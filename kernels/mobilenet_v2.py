import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig
from torch.ao.quantization.observer import default_per_channel_weight_observer
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from kernels.w3 import to_w3


# MobileNet-v2 (Sandler et al., CVPR 2018), torchvision's layout (torchvision.models.
# mobilenet_v2): a 3x3 stride-2 stem, 17 inverted-residual blocks (1x1 expand ->
# 3x3 depthwise -> 1x1 linear projection, a residual add when the block keeps its
# stride and width), a 1x1 conv to 1280 channels, global average pool, a 1000-way fc.
# Module names match torchvision's (features.N.conv.M...), so its trained weights
# load directly (the classifier's Linear is torchvision's classifier.1).
#
# One change from torchvision's float model: ReLU instead of ReLU6, as in
# torchvision's own quantizable MobileNet-v2 (torchvision.models.quantization): with
# ReLU6 the int8 conversion breaks (top-1 0%), with ReLU every activation fuses into
# its conv + BatchNorm.
NUM_CLASSES = 1000
IMAGE_SIZE = 224
BATCH_SIZE = 1
INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)

# (expansion t, output channels c, repeats n, first stride s), Sandler et al. Table 2
INVERTED_RESIDUAL_SETTING = [
    (1, 16, 1, 1), (6, 24, 2, 2), (6, 32, 3, 2), (6, 64, 4, 2),
    (6, 96, 3, 1), (6, 160, 3, 2), (6, 320, 1, 1),
]
STEM_CHANNELS = 32
LAST_CHANNELS = 1280


def _conv_bn_relu(in_ch: int, out_ch: int, kernel_size: int, stride: int, groups: int) -> nn.Sequential:
    """torchvision's Conv2dNormActivation layout (0 conv, 1 bn, 2 activation), with ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                  groups=groups, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=False),
    )


class InvertedResidual(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, stride: int, expand_ratio: int):
        super().__init__()
        hidden = in_ch * expand_ratio
        self.use_res_connect = stride == 1 and in_ch == out_ch
        layers = []
        if expand_ratio != 1:
            layers.append(_conv_bn_relu(in_ch, hidden, 1, 1, 1))                 # 1x1 expand
        layers += [
            _conv_bn_relu(hidden, hidden, 3, stride, hidden),                     # 3x3 depthwise
            nn.Conv2d(hidden, out_ch, 1, bias=False),                             # 1x1 linear projection
            nn.BatchNorm2d(out_ch),
        ]
        self.conv = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)


class MobileNetV2Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        features = [_conv_bn_relu(3, STEM_CHANNELS, 3, 2, 1)]
        in_ch = STEM_CHANNELS
        for t, c, n, s in INVERTED_RESIDUAL_SETTING:
            for i in range(n):
                features.append(InvertedResidual(in_ch, c, s if i == 0 else 1, t))
                in_ch = c
        features.append(_conv_bn_relu(in_ch, LAST_CHANNELS, 1, 1, 1))
        self.features = nn.Sequential(*features)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(LAST_CHANNELS, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x).flatten(1)
        return self.classifier(x)


# The int8 mobilenet-v2: built once by run_scripts/measure_accuracy/build_mobilenet_v2_int8.py
# (real torchvision MobileNet-v2 weights, calibrated on real ImageNet images) and
# loaded from here by the compiler.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "mobilenet_v2_int8.pt")
QUANTIZED_ENGINE = "qnnpack"


def _prepared(model: MobileNetV2Kernel, example_input: torch.Tensor) -> torch.fx.GraphModule:
    """`model` traced with quantize_mobilenet_v2's scheme and its observers inserted."""
    torch.backends.quantized.engine = QUANTIZED_ENGINE
    per_tensor = get_default_qconfig(QUANTIZED_ENGINE)
    weighted = QConfig(activation=per_tensor.activation, weight=default_per_channel_weight_observer)
    # BatchNorm and ReLU share the conv's qconfig: fusion needs one qconfig for the
    # whole conv+bn(+relu) group; every ReLU here follows a conv+bn
    mapping = (QConfigMapping().set_object_type(nn.Conv2d, weighted).set_object_type(nn.Linear, weighted)
               .set_object_type(nn.BatchNorm2d, weighted).set_object_type(nn.ReLU, weighted))
    return prepare_fx(model.eval(), mapping, (example_input,))


def quantize_mobilenet_v2(model: MobileNetV2Kernel, calibration_inputs: list) -> torch.fx.GraphModule:
    """The quantization scheme, the same as kernels/resnet50.py's quantize_resnet50:
    every nn.Conv2d (depthwise included) and nn.Linear to int8 by static post-training
    quantization (torch.ao quantize_fx, qnnpack: per-tensor affine activations, weights
    symmetric with one scale per output channel). Each BatchNorm is folded into its
    conv and each ReLU fused into it (the requantize clamps at the zero point). The
    residual adds and the pool stay float.
    calibration_inputs: images [BATCH_SIZE, *INPUT_SHAPE] the observers see."""
    prepared = _prepared(model, calibration_inputs[0])
    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)
    return convert_fx(prepared)


class MobileNetV2Int8(nn.Module):
    """The saved int8 mobilenet-v2 (INT8_CHECKPOINT); raises if it has not been built.
    The checkpoint is the int8 model's state dict (a pickled int8 graph with fused
    ConvReLU2d does not unpickle): the same int8 graph is rebuilt from
    MobileNetV2Kernel with quantize_mobilenet_v2's scheme, then the saved weights,
    scales and zero points are loaded into it (strict)."""

    def __init__(self):
        super().__init__()
        state = torch.load(INT8_CHECKPOINT, weights_only=False)
        example = torch.zeros(BATCH_SIZE, *INPUT_SHAPE)
        prepared = _prepared(MobileNetV2Kernel(), example)
        with torch.no_grad():
            prepared(example)
        self._quantized = convert_fx(prepared)
        self._quantized.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class MobileNetV2W3(nn.Module):
    """The int8 mobilenet-v2 (MobileNetV2Int8) with group-wise 3-bit weights
    (kernels/w3.py): int8 activations, every conv (depthwise ones included) and the
    classifier on 3-bit weights in groups of 64 rows."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(MobileNetV2Int8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
