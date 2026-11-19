import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig
from torch.ao.quantization.observer import default_per_channel_weight_observer
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from kernels.w3 import to_w3


# VGG-16 (Simonyan & Zisserman, ICLR 2015, configuration D), torchvision's layout
# (torchvision.models.vgg16): 13 3x3 conv + ReLU layers in 5 stages, each ending in a
# 2x2 max pool, then three fully connected layers (25088 -> 4096 -> 4096 -> 1000)
# with ReLU. `features` matches torchvision's module indices, so its trained weights
# load directly; the classifier's Linears are torchvision's classifier.0 / .3 / .6.
#
# Left out of torchvision's model, both exact at inference: the AdaptiveAvgPool2d
# ((7, 7)) before the classifier (a 224 x 224 input reaches it at 7 x 7 already, so it
# is the identity) and the two Dropouts (identity in eval). Each ReLU is its own
# non-inplace module, so it fuses with its conv / Linear when quantized.
NUM_CLASSES = 1000
IMAGE_SIZE = 224
BATCH_SIZE = 1
INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)

# configuration D: output channels of each 3x3 conv, "M" a 2x2 max pool
CONFIG_D = [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"]
FEATURE_SIZE = 7          # spatial size after the 5 pools at IMAGE_SIZE
HIDDEN = 4096


class VGG16Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        layers, in_ch = [], 3
        for v in CONFIG_D:
            if v == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers += [nn.Conv2d(in_ch, v, kernel_size=3, padding=1), nn.ReLU(inplace=False)]
                in_ch = v
        self.features = nn.Sequential(*layers)
        self.fc1 = nn.Linear(in_ch * FEATURE_SIZE * FEATURE_SIZE, HIDDEN)
        self.relu1 = nn.ReLU(inplace=False)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.relu2 = nn.ReLU(inplace=False)
        self.fc3 = nn.Linear(HIDDEN, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x).flatten(1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        return self.fc3(x)


# The int8 vgg16: built once by run_scripts/measure_accuracy/build_vgg16_int8.py (real
# torchvision VGG-16 weights, calibrated on real ImageNet images) and loaded from here
# by the compiler.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "vgg16_int8.pt")
QUANTIZED_ENGINE = "qnnpack"


def _prepared(model: VGG16Kernel, example_input: torch.Tensor) -> torch.fx.GraphModule:
    """`model` traced with quantize_vgg16's scheme and its observers inserted."""
    torch.backends.quantized.engine = QUANTIZED_ENGINE
    per_tensor = get_default_qconfig(QUANTIZED_ENGINE)
    weighted = QConfig(activation=per_tensor.activation, weight=default_per_channel_weight_observer)
    # the ReLUs share the conv's / Linear's qconfig: fusion needs one qconfig for the
    # whole conv+relu / linear+relu pair; every ReLU here follows a conv or a Linear
    mapping = (QConfigMapping().set_object_type(nn.Conv2d, weighted).set_object_type(nn.Linear, weighted)
               .set_object_type(nn.ReLU, weighted))
    return prepare_fx(model.eval(), mapping, (example_input,))


def quantize_vgg16(model: VGG16Kernel, calibration_inputs: list) -> torch.fx.GraphModule:
    """The quantization scheme, the same as kernels/resnet50.py's quantize_resnet50:
    every nn.Conv2d and nn.Linear to int8 by static post-training quantization
    (torch.ao quantize_fx, qnnpack: per-tensor affine activations, weights symmetric
    with one scale per output channel), each ReLU fused into its conv / Linear (the
    requantize clamps at the zero point). The max pools run on the int8 values.
    calibration_inputs: images [BATCH_SIZE, *INPUT_SHAPE] the observers see."""
    prepared = _prepared(model, calibration_inputs[0])
    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)
    return convert_fx(prepared)


class VGG16Int8(nn.Module):
    """The saved int8 vgg16 (INT8_CHECKPOINT); raises if it has not been built. The
    checkpoint is the int8 model's state dict (a pickled int8 graph with fused
    ConvReLU2d does not unpickle): the same int8 graph is rebuilt from VGG16Kernel with
    quantize_vgg16's scheme, then the saved weights, scales and zero points are loaded
    into it (strict)."""

    def __init__(self):
        super().__init__()
        state = torch.load(INT8_CHECKPOINT, weights_only=False)
        example = torch.zeros(BATCH_SIZE, *INPUT_SHAPE)
        prepared = _prepared(VGG16Kernel(), example)
        with torch.no_grad():
            prepared(example)
        self._quantized = convert_fx(prepared)
        self._quantized.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class VGG16W3(nn.Module):
    """The int8 vgg16 (VGG16Int8) with group-wise 3-bit weights (kernels/w3.py): int8
    activations, every conv and fc on 3-bit weights in groups of 64 rows."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(VGG16Int8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
