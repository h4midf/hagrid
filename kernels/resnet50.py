import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig
from torch.ao.quantization.observer import default_per_channel_weight_observer
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from kernels.w3 import to_w3


# Real ResNet50-v1.5 architecture (stem: 7x7 stride-2 conv + maxpool;
# 4 stages of bottleneck blocks with [3, 4, 6, 3] blocks and channel
# widths [64, 128, 256, 512] (x4 expansion); global average pool; 1000-
# way fc head). v1.5 specifically: the stride-2 downsampling in each
# stage's first bottleneck is on the 3x3 conv, not the first 1x1 (the
# actual v1/v1.5 distinction), matching the real MLPerf reference.
#
# This is the first of the conv-based kernels: nothing in
# CIMAbstraction.Ops handles Conv2d, BatchNorm2d, MaxPool2d, or
# AdaptiveAvgPool2d yet. That's the real gap this file exists to expose
# -- the compiler-side work (conv-as-matmul/im2col lowering, batchnorm
# folding, a max-reduction variant of reduce_vec) is exactly what "we'll
# handle one by one" refers to, not anything simplified in this file's
# architecture itself.
NUM_CLASSES = 1000
IMAGE_SIZE = 224
BATCH_SIZE = 1
INPUT_SHAPE = (3, IMAGE_SIZE, IMAGE_SIZE)


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels: int, mid_channels: int, stride: int = 1, downsample: nn.Module = None):
        super().__init__()
        out_channels = mid_channels * self.expansion
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        # v1.5: stride-2 lives on the 3x3 conv, not the 1x1.
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)
        # three ReLU modules, not one used three times (torchvision's form, same math):
        # relu1/relu2 fuse with their conv+bn when quantized, relu3 follows the
        # residual add (see quantize_resnet50)
        self.relu1 = nn.ReLU(inplace=False)
        self.relu2 = nn.ReLU(inplace=False)
        self.relu3 = nn.ReLU(inplace=False)
        self.downsample = downsample

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        return self.relu3(out + identity)


def _make_stage(in_channels: int, mid_channels: int, num_blocks: int, stride: int) -> nn.Sequential:
    out_channels = mid_channels * Bottleneck.expansion
    downsample = None
    if stride != 1 or in_channels != out_channels:
        downsample = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
            nn.BatchNorm2d(out_channels),
        )
    blocks = [Bottleneck(in_channels, mid_channels, stride=stride, downsample=downsample)]
    for _ in range(1, num_blocks):
        blocks.append(Bottleneck(out_channels, mid_channels))
    return nn.Sequential(*blocks)


class ResNet50Kernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.stem_conv = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.stem_bn = nn.BatchNorm2d(64)
        self.stem_relu = nn.ReLU(inplace=False)
        self.stem_pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.stage1 = _make_stage(64, 64, 3, stride=1)
        self.stage2 = _make_stage(256, 128, 4, stride=2)
        self.stage3 = _make_stage(512, 256, 6, stride=2)
        self.stage4 = _make_stage(1024, 512, 3, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(2048, NUM_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.stem_pool(self.stem_relu(self.stem_bn(self.stem_conv(x))))
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.avgpool(x).flatten(1)
        return self.fc(x)


# The int8 resnet50: built once by run_scripts/measure_accuracy/build_resnet50_int8.py
# (real torchvision ResNet-50 v1.5 weights, calibrated on real ImageNet images) and
# loaded from here by the compiler.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "resnet50_int8.pt")
QUANTIZED_ENGINE = "qnnpack"


def _prepared(model: ResNet50Kernel, example_input: torch.Tensor) -> torch.fx.GraphModule:
    """`model` traced with quantize_resnet50's scheme and its observers inserted."""
    torch.backends.quantized.engine = QUANTIZED_ENGINE
    per_tensor = get_default_qconfig(QUANTIZED_ENGINE)
    weighted = QConfig(activation=per_tensor.activation, weight=default_per_channel_weight_observer)
    # the BatchNorms and ReLUs share the conv's qconfig: fusion needs one qconfig for
    # the whole conv+bn(+relu) group (each BatchNorm is folded into its conv's weights);
    # relu3 (after the residual add) is not fused and stays float
    mapping = (QConfigMapping().set_object_type(nn.Conv2d, weighted).set_object_type(nn.Linear, weighted)
               .set_object_type(nn.BatchNorm2d, weighted)
               .set_object_type(nn.ReLU, weighted)
               .set_module_name_regex(r"stage\d\.\d+\.relu3", None))
    return prepare_fx(model.eval(), mapping, (example_input,))


def quantize_resnet50(model: ResNet50Kernel, calibration_inputs: list) -> torch.fx.GraphModule:
    """The quantization scheme, the same as kernels/vit.py's quantize_vit_encoder for
    weights: every nn.Conv2d and nn.Linear to int8 by static post-training quantization
    (torch.ao quantize_fx, qnnpack: per-tensor affine activations, weights symmetric
    with one scale per output channel). quantize_fx folds each BatchNorm into its conv,
    and fuses the ReLU that directly follows a conv+bn (the stem's, each bottleneck's
    relu1/relu2) into that int8 conv: its requantize clamps at the zero point. The ReLU
    after each residual add (relu3), the add, and the pools stay float.
    calibration_inputs: images [BATCH_SIZE, *INPUT_SHAPE] the observers see."""
    prepared = _prepared(model, calibration_inputs[0])
    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)
    return convert_fx(prepared)


class ResNet50Int8(nn.Module):
    """The saved int8 resnet50 (INT8_CHECKPOINT); raises if it has not been built.
    The checkpoint is the int8 model's state dict (a pickled int8 graph with fused
    ConvReLU2d does not unpickle): the same int8 graph is rebuilt from ResNet50Kernel
    with quantize_resnet50's scheme, then the saved weights, scales and zero points are
    loaded into it (strict)."""

    def __init__(self):
        super().__init__()
        state = torch.load(INT8_CHECKPOINT, weights_only=False)
        example = torch.zeros(BATCH_SIZE, *INPUT_SHAPE)
        prepared = _prepared(ResNet50Kernel(), example)
        with torch.no_grad():
            prepared(example)
        self._quantized = convert_fx(prepared)
        self._quantized.load_state_dict(state, strict=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class ResNet50W3(nn.Module):
    """The int8 resnet50 (ResNet50Int8) with group-wise 3-bit weights (kernels/w3.py):
    int8 activations, every conv and the fc on 3-bit weights in groups of 64 rows."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(ResNet50Int8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
