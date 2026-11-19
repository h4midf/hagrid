"""The gen-dialect bridge: converts a raw, freshly `torch.fx.symbolic_trace`d
`GraphModule` (before any rewriting at all) into a `gen_dialect`
`builtin.ModuleOp`, tagging every op with its fx_name. Bit widths are the
element types (the quantized model's integer types, and the fixed-point
formats compiler/xdsl_fixed_point.py assigns after the cim lowering).
"""

import builtins
import operator

import torch
import torch.ao.nn.intrinsic.quantized as _torch_intrinsic_quantized
import torch.ao.nn.quantized as _torch_quantized
import torch.nn as _torch_nn
from torch.fx.graph_module import GraphModule
from torch.fx.node import Node

from xdsl.dialects.builtin import (
    ArrayAttr,
    Float64Type,
    ModuleOp,
    StringAttr,
    TensorType,
    f32,
    i1,
    i8,
    i32,
    i64,
)
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Operation, Region

from compiler.dialects import gen_dialect as gd
from compiler.dialects.common_dialect import ConstRefOp, si8, ui8
from compiler.dialects.rewrite_helpers import shape_of
from compiler.support_strings import TENSOR_META

_TORCH_DTYPE_TO_XDSL = {
    torch.float32: f32,
    torch.float64: Float64Type(),
    torch.int64: i64,
    torch.int32: i32,
    torch.int8: i8,
    torch.bool: i1,
    torch.qint8: si8,
    torch.quint8: ui8,
}


def _xdsl_element_type(torch_dtype, what: str):
    """The xDSL element type of a torch dtype; raises for a dtype with no mapping."""
    if torch_dtype not in _TORCH_DTYPE_TO_XDSL:
        raise NotImplementedError(f"{what}: torch dtype {torch_dtype} has no xDSL element type")
    return _TORCH_DTYPE_TO_XDSL[torch_dtype]


def tensor_type_for(node: Node) -> TensorType:
    tm = node.meta.get(TENSOR_META)
    if tm is None or getattr(tm, "shape", None) is None:
        raise ValueError(f"{node.name}: no tensor_meta -- run ShapeProp first")
    shape = [int(d) for d in tuple(tm.shape)]
    return TensorType(_xdsl_element_type(tm.dtype, node.name), shape)


_SCALE_ZERO_POINT_ARG_POSITIONS = {
    torch.quantize_per_tensor: (1, 2),
    torch.ops.quantized.matmul: (2, 3),
}


def _is_scale_or_zero_point_only(node: Node) -> bool:
    """True if every user of this get_attr node takes it only as a
    quantize_per_tensor / quantized.matmul scale or zero point. Those are resolved
    to numbers (resolve_scalar) and stored on the op, so they get no const.ref."""
    if not node.users:
        return False
    for user in node.users:
        positions = _SCALE_ZERO_POINT_ARG_POSITIONS.get(user.target)
        if positions is None:
            return False
        if not any(len(user.args) > i and user.args[i] is node for i in positions):
            return False
    return True


# kernels/gpt2.py's int8 matmuls (torch.fx.wrap leaves, matched by qualified name): the
# raw forms of gen.quantized_linear / gen.quantized_matmul, an int32 accumulator result
# (their dequantization is a separate op)
_INT8_LINEAR = "kernels.gpt2.int8_linear"
_INT8_MATMUL = "kernels.gpt2.int8_matmul"
# kernels/gpt2.py's per-token int8 quantisation (also torch.fx.wrap leaves)
_ROW_ABSMAX = "kernels.gpt2.row_absmax"
_QUANTIZE_PER_TOKEN = "kernels.gpt2.quantize_per_token"
_DEQUANTIZE_PER_TOKEN = "kernels.gpt2.dequantize_per_token"
# kernels/gpt2.py's static per-tensor int8 quantisation (V before P.V): symmetric, scale
# x_max / 127, zero point 0 -> the existing gen.quantize_per_tensor
_QUANTIZE_PER_TENSOR_STATIC = "kernels.gpt2.quantize_per_tensor_static"
_SYMMETRIC_INT8_MAX = 127
_WHERE = "torch.where"
# bits of a qint8 weight (torch's quantized Linear / Conv, kernels/gpt2.py's int8_linear)
_QINT8_BITS = 8
# kernels/w3.py's group-wise low-bit Linear / conv (torch.fx.wrap leaves): stored
# weight codes and one scale per group of input rows, int8 activations in and out
_W3_LINEAR = "kernels.w3.w3_linear"
_W3_CONV2D = "kernels.w3.w3_conv2d"
# kernels/gpt2.py's group-wise low-bit form of int8_linear: per-token int8 activations,
# the groups' accumulators rescaled and summed (dequantize_per_token follows)
_W3_LINEAR_RAW = "kernels.gpt2.w3_linear_raw"
# their buffer arguments (codes, group scales or factors, bias): named on the op, no
# const.ref
_W3_BUFFER_POSITIONS = {_W3_LINEAR: (1, 2, 3), _W3_CONV2D: (1, 2, 3), _W3_LINEAR_RAW: (1, 2)}

# function -> its argument positions that may be a 0-d stored buffer used as a number
# (resolved onto the op like quantize_per_tensor's scale, so no const.ref)
_SCALAR_ARG_POSITIONS = {
    _DEQUANTIZE_PER_TOKEN: (2,),          # col_max: V's calibrated maximum (P.V)
    _QUANTIZE_PER_TENSOR_STATIC: (1,),    # x_max: V's calibrated maximum
    _WHERE: (2,),                         # other: e.g. GPT-2's causal-mask fill value
}


def _qualified_name(target) -> str:
    return f"{getattr(target, '__module__', '')}.{getattr(target, '__name__', '')}"


def _is_0d_buffer(gm: GraphModule, node) -> bool:
    if not (isinstance(node, Node) and node.op == "get_attr"):
        return False
    value = _resolve_attr_path(gm, node.target)
    return torch.is_tensor(value) and value.dim() == 0


# kernels/gpt2.py's wrapped functions (and torch.where) -> the config kind of their node
# (variant_generators.discover_alt_ops): the int8 matmuls take the linear / matmul
# entries (their tile reduce); the others one circuit each, so their entry is the kind
# and data_width only
_FUNCTION_KINDS = {
    _INT8_LINEAR: "linear",
    _W3_LINEAR: "linear",
    _W3_LINEAR_RAW: "linear",
    # a conv lowers to the same weighted matmul as a Linear
    _W3_CONV2D: "linear",
    _INT8_MATMUL: "matmul",
    _ROW_ABSMAX: "row_absmax",
    _QUANTIZE_PER_TOKEN: "quantize_per_token",
    _DEQUANTIZE_PER_TOKEN: "dequantize_per_token",
    _WHERE: "where",
}


def function_config_kind(node: Node) -> str | None:
    """The config kind of a call_function node the frontend maps by name (kernels/gpt2.py's
    int8 functions, torch.where), None for any other node."""
    if node.op != "call_function":
        return None
    return _FUNCTION_KINDS.get(_qualified_name(node.target))


def is_quantize_per_tensor_static(node: Node) -> bool:
    """True for a call of kernels/gpt2.py's quantize_per_tensor_static."""
    return node.op == "call_function" and _qualified_name(node.target) == _QUANTIZE_PER_TENSOR_STATIC


def static_quantize_params(gm: GraphModule, node: Node) -> tuple:
    """(scale, zero_point) of a quantize_per_tensor_static(x, x_max) call: symmetric int8,
    scale x_max / 127, zero point 0. x_max must be a 0-d stored buffer or a number. The
    frontend (gen.quantize_per_tensor) and variant_generators (its config entry) both
    read it here, so they agree."""
    x_max = node.args[1]
    if isinstance(x_max, Node) and not _is_0d_buffer(gm, x_max):
        raise NotImplementedError(f"{node.name}: quantize_per_tensor_static's x_max must be a "
                                  "0-d stored buffer or a number (a static scale)")
    return float(resolve_scalar(gm, x_max)) / _SYMMETRIC_INT8_MAX, 0


def _is_wrapped_scalar_arg_only(gm: GraphModule, node: Node) -> bool:
    """True if this get_attr node is a 0-d buffer and every user takes it only at one of
    its _SCALAR_ARG_POSITIONS: it is resolved to a number on those ops, no const.ref."""
    if not node.users or not _is_0d_buffer(gm, node):
        return False
    for user in node.users:
        positions = _SCALAR_ARG_POSITIONS.get(_qualified_name(user.target)) if user.op == "call_function" else None
        if positions is None:
            return False
        uses = [i for i, a in enumerate(user.args) if a is node]
        if not uses or any(i not in positions for i in uses):
            return False
    return True


def _is_int8_linear_weight_only(node: Node) -> bool:
    """True if every user of this get_attr node takes it only as int8_linear's weight
    (its second argument). The op names the weight, as gen.quantized_linear does a
    quantized Linear's, so it gets no const.ref."""
    return bool(node.users) and all(
        user.op == "call_function" and _qualified_name(user.target) == _INT8_LINEAR
        and len(user.args) > 1 and user.args[1] is node and user.args[0] is not node
        for user in node.users)


def _is_w3_buffer_only(node: Node) -> bool:
    """True if every user of this get_attr node takes it only as a w3_linear / w3_conv2d
    buffer (codes, group scales, bias): the op names them, so they get no const.ref."""
    return bool(node.users) and all(
        user.op == "call_function" and _qualified_name(user.target) in _W3_BUFFER_POSITIONS
        and all(i in _W3_BUFFER_POSITIONS[_qualified_name(user.target)]
                for i, a in enumerate(user.args) if a is node)
        for user in node.users)


def is_w3_call(node: Node) -> bool:
    """True for a call of kernels/w3.py's w3_linear or w3_conv2d (requantized to int8)."""
    return node.op == "call_function" and _qualified_name(node.target) in (_W3_LINEAR, _W3_CONV2D)


def w3_output_quantization(node: Node) -> tuple[float, int]:
    """(output scale, output zero point) of a w3_linear / w3_conv2d call: its last two
    arguments. The frontend and variant_generators both read them here."""
    return float(node.args[-2]), int(node.args[-1])


def _w3_op(gm: GraphModule, node: Node, v, rt: TensorType):
    """kernels/w3.py's w3_linear / w3_conv2d -> gen.quantized_linear / gen.quantized_conv2d
    with weight_bits and group_rows from the call, the codes / group scales / bias named
    on the op, and combined_scale_name the [groups, out] input scale x group scale.
      w3_linear(x, codes, group_scale, bias, weight_bits, group_rows, relu, out_scale, out_zp)
      w3_conv2d(x, codes, group_scale, bias, weight_bits, group_rows, stride, padding,
                dilation, groups, relu, out_scale, out_zp)
    kernels/gpt2.py's w3_linear_raw -> the group-rescaled raw gen.quantized_linear, its
    [groups, out] factors named combined_scale_name:
      w3_linear_raw(q, codes, group_factor, weight_bits, group_rows)"""
    name = _qualified_name(node.target)
    if name == _W3_LINEAR_RAW:
        codes, group_factor = node.args[1:3]
        for buf in (codes, group_factor):
            if not (isinstance(buf, Node) and buf.op == "get_attr"):
                raise NotImplementedError(f"{node.name}: w3_linear_raw's codes and group factors must be stored "
                                          "buffers")
        return gd.GenQuantizedLinearOp(v(node.args[0]), str(codes.target).rsplit(".", 1)[0], str(codes.target),
                                       int(node.args[3]), int(node.args[4]), None, str(group_factor.target),
                                       None, None, rt)
    codes, group_scale, bias = node.args[1:4]
    for buf in (codes, group_scale) if bias is None else (codes, group_scale, bias):
        if not (isinstance(buf, Node) and buf.op == "get_attr"):
            raise NotImplementedError(f"{node.name}: {name}'s codes, group scales and bias must be stored buffers")
    weight_name = str(codes.target)
    target = weight_name.rsplit(".", 1)[0]
    bias_name = None if bias is None else str(bias.target)
    weight_bits, group_rows = int(node.args[4]), int(node.args[5])
    relu = bool(node.args[-3])
    out_scale, out_zp = w3_output_quantization(node)
    if relu and out_zp != 0:
        # the fused ReLU is the requantize clamp at zero point 0
        raise NotImplementedError(f"{node.name}: fused ReLU with output zero point {out_zp}")
    combined_scale_name = f"{target}.combined_scale"
    if name == _W3_LINEAR:
        return gd.GenQuantizedLinearOp(v(node.args[0]), target, weight_name, weight_bits, group_rows,
                                       None, combined_scale_name, out_scale, out_zp, rt, bias_name=bias_name)
    stride, padding, dilation, groups = node.args[6:10]
    # codes [out_channels, in_channels / groups, kH, kW]
    kernel_size = tuple(int(d) for d in _resolve_attr_path(gm, codes.target).shape[2:])
    return gd.GenQuantizedConv2dOp(v(node.args[0]), target, weight_name, weight_bits, group_rows,
                                   kernel_size, _pair(dilation), _pair(padding), _pair(stride), int(groups),
                                   None, combined_scale_name, out_scale, out_zp, rt, bias_name=bias_name)


def resolve_scalar(gm: GraphModule, arg) -> float | int:
    """The number behind a quantized op's scale / zero_point argument: a literal, or
    a get_attr node naming a 0-d tensor buffer (how quantize_fx stores them)."""
    value = _resolve_attr_path(gm, arg.target) if isinstance(arg, Node) else arg
    return value.item() if torch.is_tensor(value) else value


# F.elu(input, alpha=1.0, inplace=False): torch's own value when the call omits alpha
_TORCH_ELU_ALPHA = 1.0


def elu_alpha(node: Node) -> float:
    """The alpha of an F.elu call node, as torch evaluates it."""
    if len(node.args) > 1:
        return float(node.args[1])
    if "alpha" in node.kwargs:
        return float(node.kwargs["alpha"])
    return _TORCH_ELU_ALPHA


# F.gelu(input, approximate="none"): torch's own default; "none" is the erf form, "tanh"
# the tanh form (GPT-2's gelu_new). Different functions, costed on different circuits
# (compiler/ops/gelu.py), so the form is recorded on the op, never dropped.
_TORCH_GELU_APPROXIMATE = "none"
_GELU_FORMS = ("none", "tanh")


def _gelu_form(form: str, where: str) -> str:
    if form not in _GELU_FORMS:
        raise ValueError(f"{where}: GELU approximate={form!r}, expected one of {_GELU_FORMS}")
    return form


def gelu_approximate(node: Node) -> str:
    """The GELU form of an F.gelu call node, as torch evaluates it: "none" or "tanh"."""
    if len(node.args) > 1:
        return _gelu_form(node.args[1], node.name)
    return _gelu_form(node.kwargs.get("approximate", _TORCH_GELU_APPROXIMATE), node.name)


def _with_gelu_form(op: Operation, form: str) -> Operation:
    """Records the GELU form on a gen GELU op (attribute "approximate")."""
    op.attributes["approximate"] = StringAttr(form)
    return op


def _resolve_attr_path(gm: GraphModule, target: str):
    """The object at a dotted get_attr target (e.g. "layers.0.Wq.weight")."""
    obj = gm
    for part in target.split("."):
        obj = getattr(obj, part)
    return obj


def _tensor_type_from_real_value(gm: GraphModule, node: Node) -> TensorType:
    """The type of a get_attr node with no tensor_meta, read off the buffer itself."""
    value = _resolve_attr_path(gm, node.target)
    if not torch.is_tensor(value):
        raise ValueError(f"{node.name}: get_attr target {node.target!r} is not a tensor and has no tensor_meta")
    shape = [int(d) for d in tuple(value.shape)]
    return TensorType(_xdsl_element_type(value.dtype, node.name), shape)


def _float_matmul(node: Node, *_args):
    """Matmuls must be the quantized model's int8 ops (quantized Linear,
    quantized.matmul); a float matmul, Linear or conv means the model is not quantized."""
    target = getattr(node.target, "__name__", node.target)
    raise NotImplementedError(f"{node.name}: float matmul/Linear/conv ({target}) -- only the int8 quantized "
                              "model's matmuls are supported")


def _softmax_dim(node: Node) -> int:
    if len(node.args) > 1:
        return int(node.args[1])
    return int(node.kwargs["dim"])


# one builder table per fx node kind: call_function, call_method (call_module is
# _gen_call_module_forward)
_GEN_CALL_FUNCTION_FORWARD = {
    torch.matmul: _float_matmul,
    operator.add: lambda node, v, rt: gd.GenAddOp(v(node.args[0]), v(node.args[1]), rt),
    torch.add: lambda node, v, rt: gd.GenAddOp(v(node.args[0]), v(node.args[1]), rt),
    operator.mul: lambda node, v, rt: gd.GenMulOp(v(node.args[0]), v(node.args[1]), rt),
    torch.mul: lambda node, v, rt: gd.GenMulOp(v(node.args[0]), v(node.args[1]), rt),
    operator.truediv: lambda node, v, rt: gd.GenDivOp(v(node.args[0]), v(node.args[1]), rt),
    torch.div: lambda node, v, rt: gd.GenDivOp(v(node.args[0]), v(node.args[1]), rt),
    torch.sqrt: lambda node, v, rt: gd.GenSqrtOp(v(node.args[0]), rt),
    # neg is a multiply by -1.0
    operator.neg: lambda node, v, rt: gd.GenMulOp(v(node.args[0]), -1.0, rt),
    torch.neg: lambda node, v, rt: gd.GenMulOp(v(node.args[0]), -1.0, rt),
    # torch.cat's first arg is a list of nodes
    torch.cat: lambda node, v, rt: gd.GenCatOp(
        [v(t) for t in node.args[0]], rt,
        dim=int(node.kwargs.get("dim", node.args[1] if len(node.args) > 1 else 0)),
    ),
    torch.zeros_like: lambda node, v, rt: gd.GenZerosLikeOp(v(node.args[0]), rt),
    # the function form of .transpose (the call_method form is below): same op
    torch.transpose: lambda node, v, rt: gd.GenTransposeOp(v(node.args[0]), int(node.args[1]), int(node.args[2]), rt),
    torch.nn.functional.gelu: lambda node, v, rt: _with_gelu_form(gd.GenGeluOp(v(node.args[0]), rt),
                                                                  gelu_approximate(node)),
    torch.nn.functional.silu: lambda node, v, rt: gd.GenSiluOp(v(node.args[0]), rt),
    torch.nn.functional.elu: lambda node, v, rt: gd.GenEluOp(v(node.args[0]), elu_alpha(node), rt),
    torch.softmax: lambda node, v, rt: gd.GenSoftmaxOp(v(node.args[0]), rt, dim=_softmax_dim(node)),
}

_GEN_CALL_METHOD_FORWARD = {
    # v(a) before int(): a shape arg can be a node already resolved to an int (a
    # dim read off another tensor's .shape)
    "view": lambda node, v, rt: gd.GenViewOp(v(node.args[0]), [int(v(a)) for a in node.args[1:]], rt),
    "permute": lambda node, v, rt: gd.GenPermuteOp(v(node.args[0]), [int(v(a)) for a in node.args[1:]], rt),
    "reshape": lambda node, v, rt: gd.GenReshapeOp(v(node.args[0]), [int(v(a)) for a in node.args[1:]], rt),
    "transpose": lambda node, v, rt: gd.GenTransposeOp(v(node.args[0]), int(node.args[1]), int(node.args[2]), rt),
    "contiguous": lambda node, v, rt: gd.GenContiguousOp(v(node.args[0]), rt),
    "flatten": lambda node, v, rt: gd.GenFlattenOp(
        v(node.args[0]),
        int(node.args[1]) if len(node.args) > 1 else int(node.kwargs.get("start_dim", 0)),
        int(node.args[2]) if len(node.args) > 2 else int(node.kwargs.get("end_dim", -1)),
        rt,
    ),
    # .norm(p, dim=..., keepdim=...): p positional, dim and keepdim required kwargs
    "norm": lambda node, v, rt: gd.GenNormOp(
        v(node.args[0]), int(node.args[1]), int(node.kwargs["dim"]), bool(node.kwargs["keepdim"]), rt
    ),
    "unsqueeze": lambda node, v, rt: gd.GenUnsqueezeOp(v(node.args[0]), int(v(node.args[1])), rt),
    "expand": lambda node, v, rt: gd.GenBroadcastOp(v(node.args[0]), [int(v(a)) for a in node.args[1:]], rt),
    "reshape_as": lambda node, v, rt: gd.GenReshapeAsOp(v(node.args[0]), v(node.args[1]), rt),
    # .mean(dim): keepdim=False only (GenMeanOp)
    "mean": lambda node, v, rt: gd.GenMeanOp(
        v(node.args[0]), int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["dim"]), rt
    ),
    # .scatter(dim, index, src), all positional
    "scatter": lambda node, v, rt: gd.GenScatterOp(
        v(node.args[0]), v(node.args[2]), v(node.args[3]), rt, dim=int(node.args[1])
    ),
}


_SCALE_PASSTHROUGH_OPS = (
    gd.GenViewOp, gd.GenPermuteOp, gd.GenReshapeOp, gd.GenReshapeAsOp, gd.GenTransposeOp,
    gd.GenBroadcastOp, gd.GenUnsqueezeOp, gd.GenFlattenOp, gd.GenSliceOp, gd.GenChunkOp,
    # max pooling on int8 values picks one of them: same scale and zero point
    gd.GenMaxPool2dOp,
)


def _require_requantized(owner) -> None:
    """A raw int8 matmul (no output scale) yields an int32 accumulator, which has no
    single quantization scale to read."""
    if owner.output_scale is None:
        raise NotImplementedError(f"{owner.name}: a raw int8 matmul's result is an int32 accumulator "
                                  "with no quantization scale; dequantize it explicitly")


def _scale_of(value) -> float:
    """The quantization scale of a quantized value, read off the op that produced
    it. A quantized op needs its input's scale (int32 accumulator * combined scale),
    which is not stored on the op itself. Shape/layout ops keep the scale, so the
    walk passes through them; a scalar multiply scales it."""
    owner = value.owner
    if isinstance(owner, gd.GenQuantizePerTensorOp):
        return float(owner.scale.value.data)
    if isinstance(owner, (gd.GenQuantizedLinearOp, gd.GenQuantizedConv2dOp, gd.GenQuantizedMatmulOp)):
        _require_requantized(owner)
        return float(owner.output_scale.value.data)
    if isinstance(owner, gd.GenMulOp) and owner.rhs_scalar is not None:
        return _scale_of(owner.lhs) * float(owner.rhs_scalar.value.data)
    if isinstance(owner, _SCALE_PASSTHROUGH_OPS):
        return _scale_of(owner.operand)
    raise NotImplementedError(f"_scale_of: {owner} is not a real quantized-value producer")


def _zero_point_of(value) -> int:
    """The zero point of a quantized value, walked back like _scale_of (a scalar
    multiply keeps the zero point)."""
    owner = value.owner
    if isinstance(owner, gd.GenQuantizePerTensorOp):
        return int(owner.zero_point.value.data)
    if isinstance(owner, (gd.GenQuantizedLinearOp, gd.GenQuantizedConv2dOp, gd.GenQuantizedMatmulOp)):
        _require_requantized(owner)
        return int(owner.output_zero_point.value.data)
    if isinstance(owner, gd.GenMulOp) and owner.rhs_scalar is not None:
        return _zero_point_of(owner.lhs)
    if isinstance(owner, _SCALE_PASSTHROUGH_OPS):
        return _zero_point_of(owner.operand)
    raise NotImplementedError(f"_zero_point_of: {owner} is not a real quantized-value producer")


def _pair(v_):
    """A conv / pool size attribute (an int or a tuple) as a 2-tuple."""
    return (v_, v_) if isinstance(v_, int) else tuple(v_)


def _combined_scale(node: Node, operand, weight, target: str) -> tuple:
    """(combined_scale, combined_scale_name) of an int8 Linear / conv: input scale times
    the weight scale. Per-tensor weight: (one float, None). Per-channel weight (one scale
    per output channel): (None, "<target>.combined_scale"), a [out_channels] const
    (values not carried: the compiler is timing-only, like the weight and bias consts).
    Only symmetric weights (zero point 0) are supported."""
    scheme = weight.qscheme()
    if scheme in (torch.per_tensor_affine, torch.per_tensor_symmetric):
        if int(weight.q_zero_point()) != 0:
            raise NotImplementedError(f"{node.name}: per-tensor weight with zero point "
                                      f"{int(weight.q_zero_point())}, only symmetric (0) is supported")
        return _scale_of(operand) * float(weight.q_scale()), None
    if scheme in (torch.per_channel_affine, torch.per_channel_symmetric):
        if weight.q_per_channel_axis() != 0:
            raise NotImplementedError(f"{node.name}: per-channel weight on axis {weight.q_per_channel_axis()}, "
                                      "only the output-channel axis (0) is supported")
        if bool((weight.q_per_channel_zero_points() != 0).any()):
            raise NotImplementedError(f"{node.name}: per-channel weight with nonzero zero points, "
                                      "only symmetric is supported")
        return None, f"{target}.combined_scale"
    raise NotImplementedError(f"{node.name}: weight quantization scheme {scheme} is not supported")


def _gen_call_module_forward(node: Node, v, rt: TensorType, modules: dict):
    sub = modules[node.target]
    target = str(node.target)
    if type(sub) is _torch_nn.Linear:
        _float_matmul(node)
    if isinstance(sub, _torch_nn.Embedding):
        weight_name = f"{target}.weight"
        return gd.GenEmbeddingOp(v(node.args[0]), target, weight_name, int(sub.num_embeddings), rt)
    if isinstance(sub, _torch_quantized.Linear):
        if type(sub) not in (_torch_quantized.Linear, _torch_intrinsic_quantized.LinearReLU):
            raise NotImplementedError(f"{node.name}: int8 Linear subclass {type(sub).__name__} is not supported")
        if type(sub) is _torch_intrinsic_quantized.LinearReLU and int(sub.zero_point) != 0:
            # the fused ReLU is the requantize clamp at zero point 0
            raise NotImplementedError(f"{node.name}: fused LinearReLU with output zero point {int(sub.zero_point)}")
        # weight (qint8) and bias are behind sub.weight() / sub.bias(), so the names
        # are labels, not getattr paths
        weight_name = f"{target}.weight"
        bias_name = f"{target}.bias" if sub.bias() is not None else None
        operand = v(node.args[0])
        combined_scale, combined_scale_name = _combined_scale(node, operand, sub.weight(), target)
        return gd.GenQuantizedLinearOp(
            operand, target, weight_name, _QINT8_BITS, int(sub.in_features), combined_scale, combined_scale_name,
            float(sub.scale), int(sub.zero_point), rt, bias_name=bias_name,
        )
    if type(sub) in (_torch_quantized.Conv2d, _torch_intrinsic_quantized.ConvReLU2d):
        if sub.in_channels % sub.groups or sub.out_channels % sub.groups:
            raise NotImplementedError(f"{node.name}: int8 conv with {sub.in_channels} -> {sub.out_channels} "
                                      f"channels in {sub.groups} groups")
        if sub.padding_mode != "zeros":
            raise NotImplementedError(f"{node.name}: int8 conv with padding_mode={sub.padding_mode!r}")
        if type(sub) is _torch_intrinsic_quantized.ConvReLU2d and int(sub.zero_point) != 0:
            # the fused ReLU is the requantize clamp at zero point 0
            raise NotImplementedError(f"{node.name}: fused ConvReLU2d with output zero point {int(sub.zero_point)}")
        weight_name = f"{target}.weight"
        bias_name = f"{target}.bias" if sub.bias() is not None else None
        operand = v(node.args[0])
        combined_scale, combined_scale_name = _combined_scale(node, operand, sub.weight(), target)
        kh, kw = _pair(sub.kernel_size)
        patch_rows = sub.in_channels // sub.groups * kh * kw
        return gd.GenQuantizedConv2dOp(
            operand, target, weight_name, _QINT8_BITS, patch_rows,
            _pair(sub.kernel_size), _pair(sub.dilation), _pair(sub.padding),
            _pair(sub.stride), int(sub.groups), combined_scale, combined_scale_name, float(sub.scale),
            int(sub.zero_point), rt, bias_name=bias_name,
        )
    if isinstance(sub, _torch_nn.LayerNorm):
        norm_shape = tuple(int(d) for d in tuple(sub.normalized_shape))
        if len(norm_shape) != 1:
            raise RuntimeError("gen_fx_to_xdsl: only 1D normalized_shape is supported")
        affine = bool(getattr(sub, "elementwise_affine", False))
        weight_name = f"{target}.weight" if affine else None
        bias_name = f"{target}.bias" if (affine and getattr(sub, "bias", None) is not None) else None
        return gd.GenLayerNormOp(
            v(node.args[0]), target, int(norm_shape[0]), float(sub.eps), rt,
            weight_name=weight_name, bias_name=bias_name,
        )
    if isinstance(sub, _torch_nn.ReLU):
        return gd.GenReLUOp(v(node.args[0]), target, rt)
    if isinstance(sub, _torch_nn.MaxPool2d):

        return gd.GenMaxPool2dOp(
            v(node.args[0]), target, _pair(sub.kernel_size), _pair(sub.dilation), _pair(sub.padding), _pair(sub.stride), rt
        )
    if isinstance(sub, _torch_nn.AdaptiveAvgPool2d):
        output_size = sub.output_size
        output_size = (output_size, output_size) if isinstance(output_size, int) else tuple(output_size)
        return gd.GenAdaptiveAvgPool2dOp(v(node.args[0]), target, output_size, rt)
    if type(sub) in (_torch_nn.Conv1d, _torch_nn.Conv2d, _torch_nn.Conv3d, _torch_nn.ConvTranspose3d):
        # a float conv is a float matmul (im2col + matmul)
        _float_matmul(node)
    if isinstance(sub, _torch_nn.GroupNorm):
        weight_name = f"{target}.weight" if sub.affine else None
        bias_name = f"{target}.bias" if sub.affine else None
        return gd.GenGroupLikeNormOp(
            v(node.args[0]), target, int(sub.num_groups), int(sub.num_channels), float(sub.eps), rt,
            weight_name=weight_name, bias_name=bias_name,
        )
    if isinstance(sub, (_torch_nn.InstanceNorm1d, _torch_nn.InstanceNorm2d, _torch_nn.InstanceNorm3d)):
        if getattr(sub, "track_running_stats", False):
            raise NotImplementedError("gen_fx_to_xdsl: InstanceNorm track_running_stats=True is not supported")
        channels = int(sub.num_features)
        weight_name = f"{target}.weight" if sub.affine else None
        bias_name = f"{target}.bias" if sub.affine else None
        return gd.GenGroupLikeNormOp(
            v(node.args[0]), target, channels, channels, float(sub.eps), rt, weight_name=weight_name, bias_name=bias_name
        )
    if isinstance(sub, _torch_nn.LeakyReLU):
        return gd.GenLeakyReluOp(v(node.args[0]), target, float(sub.negative_slope), rt)
    if isinstance(sub, _torch_nn.SiLU):
        return gd.GenSiluModuleOp(v(node.args[0]), target, rt)
    if isinstance(sub, _torch_nn.GELU):
        return _with_gelu_form(gd.GenGeluModuleOp(v(node.args[0]), target, rt),
                               _gelu_form(sub.approximate, node.name))
    if isinstance(sub, _torch_nn.Upsample):
        if sub.mode != "nearest":
            raise NotImplementedError(f"gen_fx_to_xdsl: Upsample mode={sub.mode!r} is not supported")
        if sub.scale_factor is None:
            raise NotImplementedError("gen_fx_to_xdsl: size-based Upsample (no scale_factor) is not supported")
        sf = sub.scale_factor
        sh, sw = (sf, sf) if isinstance(sf, (int, float)) else (sf[0], sf[1])
        if int(sh) != sh or int(sw) != sw:
            raise NotImplementedError("gen_fx_to_xdsl: only integer scale_factor is supported")
        return gd.GenUpsampleOp(v(node.args[0]), target, sub.mode, int(sh), int(sw), rt)
    raise NotImplementedError(f"gen_fx_to_xdsl: no builder for call_module type {type(sub).__name__!r}")


def gen_fx_to_xdsl(gm: GraphModule) -> ModuleOp:
    """The gen-dialect module of a traced, shape-propagated GraphModule, every op
    tagged with its fx_name."""
    modules = dict(gm.named_modules())
    placeholder_nodes = [n for n in gm.graph.nodes if n.op == "placeholder"]
    arg_types = [tensor_type_for(n) for n in placeholder_nodes]
    block = Block(arg_types=arg_types)
    value_map: dict[Node, object] = dict(zip(placeholder_nodes, block.args))

    def v(arg):
        return value_map[arg] if isinstance(arg, Node) else arg

    def tag(op: Operation, node: Node) -> Operation:
        op.attributes["fx_name"] = StringAttr(node.name)
        return op

    result_type = None
    for node in gm.graph.nodes:
        if node.op == "placeholder":
            continue
        if node.op == "get_attr":
            if (_is_scale_or_zero_point_only(node) or _is_int8_linear_weight_only(node)
                    or _is_w3_buffer_only(node) or _is_wrapped_scalar_arg_only(gm, node)):
                continue
            rt = tensor_type_for(node) if node.meta.get(TENSOR_META) is not None else _tensor_type_from_real_value(gm, node)
            op = tag(ConstRefOp(str(node.target), rt), node)
            block.add_op(op)
            value_map[node] = op.result
            continue
        if node.op == "output":
            ret_val = v(node.args[0])
            result_type = ret_val.type
            block.add_op(ReturnOp(ret_val))
            continue

        if node.op == "call_function" and _qualified_name(node.target) == _INT8_LINEAR:
            # args = (q, w_int8): q int8 activations, w_int8 the int8 weight [out, in] (a
            # get_attr buffer, named on the op) -> the raw gen.quantized_linear
            weight = node.args[1]
            if not (isinstance(weight, Node) and weight.op == "get_attr"):
                raise NotImplementedError(f"{node.name}: int8_linear's weight must be a stored buffer (get_attr)")
            module_target = str(weight.target).rsplit(".", 1)[0]
            in_features = int(_resolve_attr_path(gm, weight.target).shape[1])
            op = tag(gd.GenQuantizedLinearOp(v(node.args[0]), module_target, str(weight.target),
                                             _QINT8_BITS, in_features,
                                             None, None, None, None, tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and _qualified_name(node.target) in _W3_BUFFER_POSITIONS:
            op = tag(_w3_op(gm, node, v, tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and _qualified_name(node.target) == _ROW_ABSMAX:
            # args = (x,): max |x| over the last dim -> gen.row_absmax
            op = tag(gd.GenRowAbsmaxOp(v(node.args[0]), tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and _qualified_name(node.target) == _QUANTIZE_PER_TOKEN:
            # args = (x, row_max): int8 with one scale per row -> gen.quantize_per_token
            op = tag(gd.GenQuantizePerTokenOp(v(node.args[0]), v(node.args[1]), tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and _qualified_name(node.target) == _DEQUANTIZE_PER_TOKEN:
            # args = (acc, row_max, col_max[, bias]): the int32 accumulator back to float.
            # col_max: an operand (a stored per-channel vector, or computed per token), or
            # a 0-d stored buffer resolved to a number (col_max_scalar)
            col_max = node.args[2]
            col = resolve_scalar(gm, col_max) if _is_0d_buffer(gm, col_max) else v(col_max)
            bias = node.args[3] if len(node.args) > 3 else node.kwargs.get("bias")
            op = tag(gd.GenDequantizePerTokenOp(v(node.args[0]), v(node.args[1]), col, tensor_type_for(node),
                                                bias=None if bias is None else v(bias)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if is_quantize_per_tensor_static(node):
            # args = (x, x_max) -> the existing gen.quantize_per_tensor, scale x_max / 127,
            # zero point 0 (symmetric int8)
            scale, zero_point = static_quantize_params(gm, node)
            op = tag(gd.GenQuantizePerTensorOp(v(node.args[0]), scale, zero_point, tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and node.target is torch.where:
            # args = (cond, x, other): out = cond ? x : other. other: an operand, or one
            # number (a literal or a 0-d stored buffer, e.g. GPT-2's mask fill value)
            if len(node.args) != 3:
                raise NotImplementedError(f"{node.name}: torch.where needs (cond, x, other), got {len(node.args)} args")
            cond, x, other = node.args
            if not isinstance(other, Node) or _is_0d_buffer(gm, other):
                other = float(resolve_scalar(gm, other))
            else:
                other = v(other)
            op = tag(gd.GenWhereOp(v(cond), v(x), other, tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and _qualified_name(node.target) == _INT8_MATMUL:
            # args = (a, b): two int8 activations -> the raw gen.quantized_matmul
            op = tag(gd.GenQuantizedMatmulOp(v(node.args[0]), v(node.args[1]), None, None, None,
                                             tensor_type_for(node)), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and node.target is torch.quantize_per_tensor:
            # args = (tensor, scale, zero_point, dtype); the dtype is rt's element type
            scale = resolve_scalar(gm, node.args[1])
            zero_point = resolve_scalar(gm, node.args[2])
            rt = tensor_type_for(node)
            op = tag(gd.GenQuantizePerTensorOp(v(node.args[0]), float(scale), int(zero_point), rt), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_method" and str(node.target) == "dequantize":
            rt = tensor_type_for(node)
            operand = v(node.args[0])
            op = tag(gd.GenDequantizeOp(operand, _scale_of(operand), _zero_point_of(operand), rt), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and node.target is torch.ops.quantized.matmul:
            # args = (lhs, rhs, output_scale, output_zero_point): an int8 matmul of two
            # activations (attention scores, weighted sum of V)
            scale = resolve_scalar(gm, node.args[2])
            zero_point = resolve_scalar(gm, node.args[3])
            rt = tensor_type_for(node)
            lhs, rhs = v(node.args[0]), v(node.args[1])
            combined_scale = _scale_of(lhs) * _scale_of(rhs)
            op = tag(gd.GenQuantizedMatmulOp(lhs, rhs, combined_scale, float(scale), int(zero_point), rt), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_function" and node.target is torch.ops.quantized.mul:
            # args = (tensor, scalar): output scale = input scale * scalar, zero point
            # unchanged (_scale_of). Built as a plain gen.mul: on the hardware it is the
            # same pass as any scalar multiply (torch only skips it in software)
            rt = tensor_type_for(node)
            op = tag(gd.GenMulOp(v(node.args[0]), float(node.args[1]), rt), node)
            block.add_op(op)
            value_map[node] = op.result
            continue

        if node.op == "call_method" and str(node.target) == "chunk":
            # one multi-result gen.chunk; the chunk node's value is a tuple with no
            # tensor_meta, so each chunk's type comes from its getitem user
            chunks = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs.get("chunks"))
            dim = int(node.args[2]) if len(node.args) > 2 else int(node.kwargs.get("dim", 0))
            result_types = [None] * chunks
            for user in node.users:
                if user.op == "call_function" and user.target is operator.getitem:
                    result_types[user.args[1]] = tensor_type_for(user)
            if any(t is None for t in result_types):
                raise NotImplementedError(f"{node.name}: gen_fx_to_xdsl needs every gen.chunk getitem consumer present")
            op = tag(gd.GenChunkOp(v(node.args[0]), chunks, dim, result_types), node)
            block.add_op(op)
            value_map[node] = list(op.results)
            continue

        if node.op == "call_function" and node.target is torch.topk:
            # (values, indices) split into GenMaxOp + GenArgmaxOp on the same operand,
            # dim and k; types come from the getitem users (like gen.chunk). Only the
            # indices op carries the fx_name, so no two ops claim it
            k = int(node.args[1]) if len(node.args) > 1 else int(node.kwargs["k"])
            dim = int(node.args[2]) if len(node.args) > 2 else int(node.kwargs.get("dim", -1))
            result_types = [None, None]
            for user in node.users:
                if user.op == "call_function" and user.target is operator.getitem:
                    result_types[user.args[1]] = tensor_type_for(user)
            if any(t is None for t in result_types):
                raise NotImplementedError(f"{node.name}: gen_fx_to_xdsl needs every torch.topk getitem consumer present")
            values_op = gd.GenMaxOp(v(node.args[0]), dim, result_types[0], k=k)
            block.add_op(values_op)
            indices_op = tag(gd.GenArgmaxOp(v(node.args[0]), dim, result_types[1], k=k), node)
            block.add_op(indices_op)
            value_map[node] = [values_op.result, indices_op.result]
            continue

        if node.op == "call_function" and node.target is builtins.getattr and len(node.args) > 1 and node.args[1] == "shape":
            # tensor.shape: a tuple of ints from the operand's type, no op
            value_map[node] = tuple(shape_of(v(node.args[0]).type))
            continue

        if node.op == "call_function" and node.target is operator.getitem and isinstance(value_map.get(node.args[0]), (list, tuple)):
            # getitem on a gen.chunk's results or a .shape tuple: plain indexing, no op
            value_map[node] = value_map[node.args[0]][node.args[1]]
            continue

        idx1 = node.args[1] if node.op == "call_function" and node.target is operator.getitem else None
        if isinstance(idx1, tuple) and Ellipsis in idx1:
            # `...` expands to as many full `:` slices as the operand's remaining rank
            ellipsis_pos = idx1.index(Ellipsis)
            operand_rank = len(shape_of(tensor_type_for(node.args[0])))
            fill_count = operand_rank - (len(idx1) - 1)
            idx1 = idx1[:ellipsis_pos] + (slice(None, None, None),) * fill_count + idx1[ellipsis_pos + 1:]

        if isinstance(idx1, tuple) and any(isinstance(e, slice) for e in idx1):
            # tensor slicing; every bound is an int by now (a literal or a resolved
            # .shape entry). An int index drops its dim: built as a gen.slice over
            # [i, i+1) then a gen.view to the reduced shape (x[:, 0, :] is
            # x[:, 0:1, :].view(...))
            if not all(isinstance(e, (slice, int)) for e in idx1):
                raise NotImplementedError(f"{node.name}: gen_fx_to_xdsl only supports slice/int getitem indices")

            def _bound(b):
                if b is None:
                    return gd.SLICE_NONE
                return v(b) if isinstance(b, Node) else b

            idx = idx1
            starts = []
            stops = []
            steps = []
            squeeze_dims = []
            for dim_i, e in enumerate(idx):
                if isinstance(e, slice):
                    starts.append(_bound(e.start))
                    stops.append(_bound(e.stop))
                    steps.append(_bound(e.step))
                else:
                    i = _bound(e)
                    starts.append(i)
                    stops.append(i + 1)
                    steps.append(gd.SLICE_NONE)
                    squeeze_dims.append(dim_i)

            final_type = tensor_type_for(node)
            if squeeze_dims:
                intermediate_shape = list(shape_of(final_type))
                for dim_i in squeeze_dims:
                    intermediate_shape.insert(dim_i, 1)
                slice_type = TensorType(final_type.element_type, intermediate_shape)
            else:
                slice_type = final_type

            # only the final op carries the fx_name, so no two ops claim it
            slice_op = gd.GenSliceOp(v(node.args[0]), starts, stops, steps, slice_type)
            if not squeeze_dims:
                slice_op = tag(slice_op, node)
            block.add_op(slice_op)
            result_value = slice_op.result
            if squeeze_dims:
                view_op = tag(gd.GenViewOp(result_value, list(shape_of(final_type)), final_type), node)
                block.add_op(view_op)
                result_value = view_op.result
            value_map[node] = result_value
            continue

        rt = tensor_type_for(node)
        if node.op == "call_module":
            op = tag(_gen_call_module_forward(node, v, rt, modules), node)
        elif node.op == "call_method":
            builder = _GEN_CALL_METHOD_FORWARD.get(str(node.target))
            if builder is None:
                raise NotImplementedError(f"{node.name}: no gen_dialect builder for call_method {node.target!r}")
            op = tag(builder(node, v, rt), node)
        elif node.op == "call_function":
            builder = _GEN_CALL_FUNCTION_FORWARD.get(node.target)
            if builder is None:
                target_name = getattr(node.target, "__name__", repr(node.target))
                raise NotImplementedError(f"{node.name}: no gen_dialect builder for call_function {target_name!r}")
            op = tag(builder(node, v, rt), node)
        else:
            raise NotImplementedError(f"{node.name}: unsupported fx node op {node.op!r}")
        block.add_op(op)
        value_map[node] = op.result

    arg_names = [n.name for n in placeholder_nodes]
    func_type = (arg_types, [result_type] if result_type is not None else [])
    func_op = FuncOp("main", func_type, Region([block]))
    func_op.attributes["arg_names"] = ArrayAttr([StringAttr(n) for n in arg_names])
    return ModuleOp([func_op])
