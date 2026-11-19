"""Group-wise 3-bit weights for the int8 kernels: the weight format of the RRAM/SRAM
ensemble target (targets/rram-sram-ensemble), one weight per 3-bit RRAM cell.

Every quantized Linear and Conv of an int8 model (torch's quantized modules) becomes a
w3_linear / w3_conv2d with WEIGHT_BITS-bit symmetric weights and one scale per GROUP_ROWS
input rows per output column: the format of GPTQ (Frantar et al., ICLR 2023), whose 3-bit
results use one scale per group of input rows. GROUP_ROWS is the ensemble's RRAM macro
rows (64), so each macro's partial sum has its own scale.

The codes are the int8 weights rounded to WEIGHT_BITS bits per group, with no calibration
or training: the hardware model reads the format (bits, groups), never the values, and
the accuracy is not measured. Activations stay int8: each op keeps the int8 model's input
and output scales and zero points, and the dynamic matmuls (attention) are not changed.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.ao.nn import quantized as nnq
from torch.ao.nn.intrinsic import quantized as nniq

WEIGHT_BITS = 3
GROUP_ROWS = 64
# symmetric codes, zero point 0: [-CODE_MAX, CODE_MAX]
CODE_MAX = 2 ** (WEIGHT_BITS - 1) - 1

_LINEARS = (nnq.Linear, nniq.LinearReLU)
_CONVS = (nnq.Conv2d, nniq.ConvReLU2d)
_WITH_RELU = (nniq.LinearReLU, nniq.ConvReLU2d)


def _dequantized(codes: torch.Tensor, group_scale: torch.Tensor, group_rows: int) -> torch.Tensor:
    """The real weight [out, k] of codes [out, k] with group_scale [out, groups]."""
    k = codes.shape[1]
    return codes.float() * group_scale.repeat_interleave(group_rows, dim=1)[:, :k]


def _requantized(y: torch.Tensor, relu: bool, out_scale: float, out_zp: int) -> torch.Tensor:
    if relu:
        y = F.relu(y)
    return torch.quantize_per_tensor(y, out_scale, out_zp, torch.quint8)


def w3_linear(x, codes, group_scale, bias, weight_bits, group_rows, relu, out_scale, out_zp):
    """x (quint8) @ the group-wise weight^T, + bias, (ReLU,) requantized to quint8.
    codes [out, in], group_scale [out, ceil(in / group_rows)]."""
    w = _dequantized(codes, group_scale, group_rows)
    return _requantized(F.linear(x.dequantize(), w, bias), relu, out_scale, out_zp)


def w3_conv2d(x, codes, group_scale, bias, weight_bits, group_rows, stride, padding, dilation, groups,
              relu, out_scale, out_zp):
    """conv2d of x (quint8) with the group-wise weight, + bias, (ReLU,) requantized to
    quint8. codes [out, in / groups, kH, kW], grouped over each output's in / groups * kH *
    kW patch values (im2col order): group_scale [out, ceil(patch / group_rows)]."""
    w = _dequantized(codes.reshape(codes.shape[0], -1), group_scale, group_rows).reshape(codes.shape)
    y = F.conv2d(x.dequantize(), w, bias, stride, padding, dilation, groups)
    return _requantized(y, relu, out_scale, out_zp)


torch.fx.wrap("w3_linear")
torch.fx.wrap("w3_conv2d")


def group_codes(w: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(codes [out, k] int8 in [-CODE_MAX, CODE_MAX], group_scale [out, groups]) of a real
    weight [out, k]: per GROUP_ROWS rows, scale = max |w| / CODE_MAX, codes rounded."""
    out, k = w.shape
    groups = -(-k // GROUP_ROWS)
    padded = F.pad(w, (0, groups * GROUP_ROWS - k)).reshape(out, groups, GROUP_ROWS)
    scale = padded.abs().amax(dim=2).clamp(min=torch.finfo(torch.float32).tiny) / CODE_MAX
    codes = torch.round(padded / scale[:, :, None]).clamp(-CODE_MAX, CODE_MAX)
    return codes.reshape(out, -1)[:, :k].to(torch.int8), scale


class _W3Weighted(nn.Module):
    """The stored part of a w3 op, from a torch quantized Linear / Conv: its weight as
    codes and group scales, its float bias, output scale and zero point, fused ReLU."""

    def __init__(self, sub: nn.Module):
        super().__init__()
        w = sub.weight().dequantize()
        codes, group_scale = group_codes(w.reshape(w.shape[0], -1))
        self.register_buffer("codes", codes.reshape(w.shape))
        self.register_buffer("group_scale", group_scale)
        bias = sub.bias()
        if bias is None:
            self.bias = None
        else:
            self.register_buffer("bias", bias.detach().float().clone())
        self.relu = type(sub) in _WITH_RELU
        self.out_scale = float(sub.scale)
        self.out_zp = int(sub.zero_point)


class W3Linear(_W3Weighted):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w3_linear(x, self.codes, self.group_scale, self.bias, WEIGHT_BITS, GROUP_ROWS, self.relu,
                         self.out_scale, self.out_zp)


class W3Conv2d(_W3Weighted):
    def __init__(self, sub: nn.Module):
        super().__init__(sub)
        if sub.padding_mode != "zeros":
            raise NotImplementedError(f"w3 conv with padding_mode={sub.padding_mode!r}")
        self.stride = tuple(sub.stride)
        self.padding = tuple(sub.padding)
        self.dilation = tuple(sub.dilation)
        self.groups = int(sub.groups)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return w3_conv2d(x, self.codes, self.group_scale, self.bias, WEIGHT_BITS, GROUP_ROWS, self.stride,
                         self.padding, self.dilation, self.groups, self.relu, self.out_scale, self.out_zp)


def to_w3(int8_model: nn.Module) -> nn.Module:
    """int8_model with every torch quantized Linear and Conv replaced by its w3 form, in
    place. Raises on any other module with weights it cannot convert."""
    for name, sub in list(int8_model.named_modules()):
        if type(sub) in _LINEARS:
            replacement = W3Linear(sub)
        elif type(sub) in _CONVS:
            replacement = W3Conv2d(sub)
        elif isinstance(sub, (nnq.Linear, nnq.Conv1d, nnq.Conv2d, nnq.Conv3d, nnq.ConvTranspose2d)):
            raise NotImplementedError(f"{name}: quantized module {type(sub).__name__} has no w3 form")
        else:
            continue
        parent_name, _, child = name.rpartition(".")
        setattr(int8_model.get_submodule(parent_name) if parent_name else int8_model, child, replacement)
    return int8_model
