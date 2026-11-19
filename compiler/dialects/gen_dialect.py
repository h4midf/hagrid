"""The `gen` dialect: the traced torch graph (fx call_module / call_method /
call_function nodes) one op per node, before it is lowered to `cinm_dialect`."""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, IntegerAttr, StringAttr
from xdsl.ir import Dialect, SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)

from compiler.dialects.attr_helpers import bool_prop, float_prop, int_list_prop, int_prop
from compiler.dialects.attr_helpers import is_group_rescaled_raw as _is_group_rescaled_raw
from compiler.dialects.attr_helpers import is_raw_matmul as _is_raw_matmul
from compiler.dialects.common_dialect import SLICE_NONE  # re-exported for xdsl_bridge.py (gd.SLICE_NONE)
from compiler.dialects.op_builders import binary_op_either_scalar, binary_op_scalar_rhs, unary_op


def _check_weight_format(where: str, weight_bits: int, group_rows: int) -> None:
    """A stored weight's bits per element and the input rows per weight scale."""
    if weight_bits < 1:
        raise ValueError(f"{where}: weight_bits {weight_bits}")
    if group_rows < 1:
        raise ValueError(f"{where}: group_rows {group_rows}")


GenContiguousOp = unary_op("gen.contiguous")
GenSqrtOp = unary_op("gen.sqrt")
GenGeluOp = unary_op("gen.gelu")
GenSiluOp = unary_op("gen.silu")
# kernels/gpt2.py's row_absmax: max |x| over the last dim, [..., n] -> [..., 1] (the row
# maximum of gen.quantize_per_token)
GenRowAbsmaxOp = unary_op("gen.row_absmax")


@irdl_op_definition
class GenEluOp(IRDLOperation):
    """torch.nn.functional.elu, with the call's alpha."""

    name = "gen.elu"
    operand = operand_def()
    alpha = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, alpha: float, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"alpha": float_prop(alpha)})


@irdl_op_definition
class GenChunkOp(IRDLOperation):
    """x.chunk(chunks, dim) (e.g. RoPE's rotate-half split), the dialect's one
    multi-result op: one result per chunk (fx's getitem per chunk maps to results[i])."""

    name = "gen.chunk"
    operand = operand_def()
    chunks = prop_def(IntegerAttr)
    dim = prop_def(IntegerAttr)
    outputs = var_result_def()

    def __init__(self, operand, chunks: int, dim: int, result_types: list):
        super().__init__(
            operands=[operand],
            result_types=[result_types],
            properties={"chunks": int_prop(chunks), "dim": int_prop(dim)},
        )


@irdl_op_definition
class GenCatOp(IRDLOperation):
    """torch.cat(tensors, dim)."""

    name = "gen.cat"
    tensors = var_operand_def()
    dim = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, tensors, result_type, dim: int = 0):
        super().__init__(operands=[tensors], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class GenMeanOp(IRDLOperation):
    """x.mean(dim), keepdim=False (lowered by xdsl_gen_lowering.py's MeanAsCinmPattern)."""

    name = "gen.mean"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, dim: int, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class GenArgmaxOp(IRDLOperation):
    """torch.argmax(a, dim) (k = 1), or the indices of torch.topk(a, k, dim): the bridge
    splits a topk into this op and a GenMaxOp on the same operand, dim and k."""

    name = "gen.argmax"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, dim: int, result_type, k: int):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim), "k": int_prop(k)})


@irdl_op_definition
class GenMaxOp(IRDLOperation):
    """The k largest values along dim (the values of a torch.topk, see GenArgmaxOp)."""

    name = "gen.max"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, dim: int, result_type, k: int):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim), "k": int_prop(k)})


@irdl_op_definition
class GenZerosLikeOp(IRDLOperation):
    """torch.zeros_like(x); xdsl_gen_lowering.py's ScatterAsCinmPattern fuses it with the
    gen.scatter it feeds into one cinm.scatter."""

    name = "gen.zeros_like"
    operand = operand_def()
    result = result_def()

    def __init__(self, operand, result_type):
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class GenScatterOp(IRDLOperation):
    """base.scatter(dim, index, src), with its base (cinm / cim drop it: a zeros_like)."""

    name = "gen.scatter"
    base = operand_def()
    index = operand_def()
    src = operand_def()
    dim = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, base, index, src, result_type, dim: int = -1):
        super().__init__(operands=[base, index, src], result_types=[result_type], properties={"dim": int_prop(dim)})


# x + y; a scalar may be on either side (1.0 + scale); AddAsCinmPattern moves it to
# the right
GenAddOp = binary_op_either_scalar("gen.add")
# x * y, a scalar only on the right; also torch.ops.quantized.mul(q, scalar), whose
# output scale (input scale x scalar) xdsl_bridge.py's _scale_of recovers from it
GenMulOp = binary_op_scalar_rhs("gen.mul")
# x / y; a scalar may be on either side (1.0 / std)
GenDivOp = binary_op_either_scalar("gen.div")


@irdl_op_definition
class GenEmbeddingOp(IRDLOperation):
    """nn.Embedding: operand is the integer index tensor (token ids). num_embeddings (the
    table's rows) is a property because no operand or result shape gives it; the
    embedding dim is the result's last dim."""

    name = "gen.embedding"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    weight_name = prop_def(StringAttr)
    num_embeddings = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, weight_name: str, num_embeddings: int, result_type):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={
                "module_target": StringAttr(module_target),
                "weight_name": StringAttr(weight_name),
                "num_embeddings": int_prop(num_embeddings),
            },
        )


@irdl_op_definition
class GenQuantizePerTensorOp(IRDLOperation):
    """torch.quantize_per_tensor(x, scale, zero_point, dtype): float to int8. The dtype
    (quint8 / qint8) is the result's element type (ui8 / si8)."""

    name = "gen.quantize_per_tensor"
    operand = operand_def()
    scale = prop_def(FloatAttr)
    zero_point = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, scale: float, zero_point: int, result_type):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={"scale": float_prop(scale), "zero_point": int_prop(zero_point)},
        )


@irdl_op_definition
class GenDequantizeOp(IRDLOperation):
    """The `.dequantize()` call_method: int8 -> real. scale / zero_point are the
    quantized value's own (its producer's output scale and zero point), resolved by
    the bridge, so the op carries them like quantize_per_tensor does."""

    name = "gen.dequantize"
    operand = operand_def()
    result = result_def()
    scale = prop_def(FloatAttr)
    zero_point = prop_def(IntegerAttr)

    def __init__(self, operand, scale: float, zero_point: int, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"scale": float_prop(scale), "zero_point": int_prop(zero_point)},
        )


@irdl_op_definition
class GenQuantizedLinearOp(IRDLOperation):
    """torch.ao.nn.quantized.Linear: int8 x int8 matmul, bias add, requantize to int8.
    combined_scale (input scale x weight scale; the input's from its producer, see
    xdsl_bridge.py's _scale_of) takes the int32 accumulator to float before the bias
    and the requantize (xdsl_lowering.py's QuantizedMatmulAsCimPattern). bias_name names
    a float32 const. Exactly one of combined_scale (a per-tensor weight) and
    combined_scale_name (a per-channel weight: a [out_features] const of
    input scale x weight_scale[c]).
    Raw form (no combined scale, output scale, output zero point or bias): the int8 x
    int8 matmul alone, its result the int32 accumulator (kernels/gpt2.py's int8_linear,
    whose dequantization is a separate op).
    `weight_bits`: bits per weight (8 for a qint8 weight, 3 for kernels/w3.py's).
    `group_rows`: input rows sharing one weight scale. in_features for a per-tensor or
    per-channel weight (one group); fewer for a group-wise weight (kernels/w3.py), whose
    combined_scale_name then names a [groups, out_features] const.
    Group-rescaled raw form (a group-wise weight, combined_scale_name only: no combined
    scale, output scale, output zero point or bias): each group's int accumulator times
    its [groups, out_features] factor, summed over the groups, a float result whose
    dequantization is a separate op (kernels/gpt2.py's w3_linear_raw, then
    dequantize_per_token)."""

    name = "gen.quantized_linear"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    weight_name = prop_def(StringAttr)
    weight_bits = prop_def(IntegerAttr)
    group_rows = prop_def(IntegerAttr)
    bias_name = opt_prop_def(StringAttr)
    combined_scale = opt_prop_def(FloatAttr)
    combined_scale_name = opt_prop_def(StringAttr)
    output_scale = opt_prop_def(FloatAttr)
    output_zero_point = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(
        self,
        operand,
        module_target: str,
        weight_name: str,
        weight_bits: int,
        group_rows: int,
        combined_scale: float | None,
        combined_scale_name: str | None,
        output_scale: float | None,
        output_zero_point: int | None,
        result_type,
        bias_name: str | None = None,
    ):
        _check_weight_format(f"gen.quantized_linear {module_target}", weight_bits, group_rows)
        props = {
            "module_target": StringAttr(module_target),
            "weight_name": StringAttr(weight_name),
            "weight_bits": int_prop(weight_bits),
            "group_rows": int_prop(group_rows),
        }
        if _is_raw_matmul(combined_scale, combined_scale_name, output_scale, output_zero_point, bias_name):
            super().__init__(operands=[operand], result_types=[result_type], properties=props)
            return
        if _is_group_rescaled_raw(combined_scale, combined_scale_name, output_scale, output_zero_point, bias_name):
            if group_rows >= operand.type.get_shape()[-1]:
                raise ValueError(f"gen.quantized_linear {module_target}: the group-rescaled raw form needs "
                                 f"more than one group of {group_rows} rows")
            props["combined_scale_name"] = StringAttr(combined_scale_name)
            super().__init__(operands=[operand], result_types=[result_type], properties=props)
            return
        if (combined_scale is None) == (combined_scale_name is None):
            raise ValueError(f"gen.quantized_linear {module_target}: needs exactly one of combined_scale "
                             f"(per-tensor) and combined_scale_name (per-channel)")
        if output_scale is None or output_zero_point is None:
            raise ValueError(f"gen.quantized_linear {module_target}: a requantizing op needs output_scale and "
                             f"output_zero_point (or none of the scales and no bias: the raw form)")
        props["output_scale"] = float_prop(output_scale)
        props["output_zero_point"] = int_prop(output_zero_point)
        if combined_scale is not None:
            props["combined_scale"] = float_prop(combined_scale)
        else:
            props["combined_scale_name"] = StringAttr(combined_scale_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class GenQuantizedConv2dOp(IRDLOperation):
    """torch.ao.nn.quantized.Conv2d (and the fused ConvReLU2d, whose ReLU is the
    requantize clamp at zero point 0) as a call_module: an int8 conv lowered as im2col +
    the int8 matmul of gen.quantized_linear (see QuantizedConv2dAsCinmPattern). With
    groups g > 1 (a depthwise conv: g = channels), each group's in_channels / g inputs
    feed its own out_channels / g outputs: a grouped matmul, one weight block per group.
    Weight [out_channels, in_channels / g, kH, kW] (qint8), float32 bias, and exactly one
    of combined_scale / combined_scale_name, as gen.quantized_linear. weight_bits and
    group_rows as gen.quantized_linear, the input rows being a group's
    in_channels / g * kH * kW patch values."""

    name = "gen.quantized_conv2d"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    weight_name = prop_def(StringAttr)
    weight_bits = prop_def(IntegerAttr)
    group_rows = prop_def(IntegerAttr)
    bias_name = opt_prop_def(StringAttr)
    kernel_size = prop_def(ArrayAttr[IntAttr])
    dilation = prop_def(ArrayAttr[IntAttr])
    padding = prop_def(ArrayAttr[IntAttr])
    stride = prop_def(ArrayAttr[IntAttr])
    groups = prop_def(IntegerAttr)
    combined_scale = opt_prop_def(FloatAttr)
    combined_scale_name = opt_prop_def(StringAttr)
    output_scale = prop_def(FloatAttr)
    output_zero_point = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, weight_name: str, weight_bits: int, group_rows: int,
                 kernel_size, dilation, padding, stride,
                 groups: int, combined_scale: float | None, combined_scale_name: str | None,
                 output_scale: float, output_zero_point: int, result_type, bias_name: str | None = None):
        if (combined_scale is None) == (combined_scale_name is None):
            raise ValueError(f"gen.quantized_conv2d {module_target}: needs exactly one of combined_scale "
                             f"(per-tensor) and combined_scale_name (per-channel)")
        if groups < 1:
            raise ValueError(f"gen.quantized_conv2d {module_target}: groups {groups}")
        _check_weight_format(f"gen.quantized_conv2d {module_target}", weight_bits, group_rows)
        props = {
            "module_target": StringAttr(module_target),
            "weight_name": StringAttr(weight_name),
            "weight_bits": int_prop(weight_bits),
            "group_rows": int_prop(group_rows),
            "kernel_size": int_list_prop(kernel_size),
            "dilation": int_list_prop(dilation),
            "padding": int_list_prop(padding),
            "stride": int_list_prop(stride),
            "groups": int_prop(groups),
            "output_scale": float_prop(output_scale),
            "output_zero_point": int_prop(output_zero_point),
        }
        if combined_scale is not None:
            props["combined_scale"] = float_prop(combined_scale)
        else:
            props["combined_scale_name"] = StringAttr(combined_scale_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class GenQuantizedMatmulOp(IRDLOperation):
    """torch.ops.quantized.matmul(lhs, rhs, output_scale, output_zero_point): an int8
    matmul of two activations (Q.K^T, P.V); combined_scale is lhs scale x rhs scale.
    Raw form (no combined scale, output scale or output zero point): the int8 x int8
    matmul alone, its result the int32 accumulator (kernels/gpt2.py's int8_matmul)."""

    name = "gen.quantized_matmul"
    lhs = operand_def()
    rhs = operand_def()
    combined_scale = opt_prop_def(FloatAttr)
    output_scale = opt_prop_def(FloatAttr)
    output_zero_point = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, lhs, rhs, combined_scale: float | None, output_scale: float | None,
                 output_zero_point: int | None, result_type):
        if _is_raw_matmul(combined_scale, None, output_scale, output_zero_point, None):
            super().__init__(operands=[lhs, rhs], result_types=[result_type])
            return
        if combined_scale is None or output_scale is None or output_zero_point is None:
            raise ValueError("gen.quantized_matmul: needs combined_scale, output_scale and output_zero_point "
                             "(or none of them: the raw form)")
        super().__init__(
            operands=[lhs, rhs],
            result_types=[result_type],
            properties={
                "combined_scale": float_prop(combined_scale),
                "output_scale": float_prop(output_scale),
                "output_zero_point": int_prop(output_zero_point),
            },
        )


@irdl_op_definition
class GenQuantizePerTokenOp(IRDLOperation):
    """kernels/gpt2.py's quantize_per_token(x, row_max): per-token (dynamic) int8
    quantisation, one scale per row, row_max / 127 (symmetric, zero point 0):
    q = clamp(round(x * 127 / row_max), -127, 127). x [..., n] float, row_max [..., 1] (the
    row's max |x|, gen.row_absmax) -> int8 [..., n]. Unlike gen.quantize_per_tensor the
    scale is an operand: it is computed at run time, one per row."""

    name = "gen.quantize_per_token"
    operand = operand_def()
    row_max = operand_def()
    result = result_def()

    def __init__(self, operand, row_max, result_type):
        super().__init__(operands=[operand, row_max], result_types=[result_type])


@irdl_op_definition
class GenDequantizePerTokenOp(IRDLOperation):
    """kernels/gpt2.py's dequantize_per_token(acc, row_max, col_max, bias): the int32
    accumulator of a raw int8 matmul back to float,
      acc * (row_max / 127) * (col_max / 127)  (+ bias).
    row_max [..., 1] is the lhs's per-token maximum (one per output row). col_max is the
    rhs's maximum, one per output column or one for all:
      a Linear   the weight's per-output-channel maxima, a stored [n] vector
      Q.K^T      K's per-token maxima, transposed [..., 1, n], computed at run time
      P.V        V's calibrated maximum, one number (col_max_scalar, as
                 gen.quantize_per_tensor stores its scale)
    Exactly one of col_max (an operand) and col_max_scalar. bias [n] (a Linear's float
    bias) is optional."""

    name = "gen.dequantize_per_token"
    irdl_options = (AttrSizedOperandSegments(),)
    operand = operand_def()
    row_max = operand_def()
    col_max = opt_operand_def()
    col_max_scalar = opt_prop_def(FloatAttr)
    bias = opt_operand_def()
    result = result_def()

    def __init__(self, operand, row_max, col_max, result_type, bias=None):
        props = {}
        if isinstance(col_max, SSAValue):
            col_operand = col_max
        else:
            col_operand = None
            props["col_max_scalar"] = float_prop(col_max)
        super().__init__(operands=[operand, row_max, col_operand, bias], result_types=[result_type],
                         properties=props)


@irdl_op_definition
class GenWhereOp(IRDLOperation):
    """torch.where(cond, x, other): elementwise select, out = cond ? x : other.
    cond is a bool tensor (e.g. GPT-2's stored causal mask), x a tensor, other an
    operand or one number (other_scalar: a literal or a 0-d stored buffer, resolved like
    gen.quantize_per_tensor's scale). Exactly one of other and other_scalar. The
    hardware is custom_op_library/vec_select (a 2-to-1 multiplexer per element)."""

    name = "gen.where"
    irdl_options = (AttrSizedOperandSegments(),)
    cond = operand_def()
    operand = operand_def()
    other = opt_operand_def()
    other_scalar = opt_prop_def(FloatAttr)
    result = result_def()

    def __init__(self, cond, operand, other, result_type):
        props = {}
        if isinstance(other, SSAValue):
            other_operand = other
        else:
            other_operand = None
            props["other_scalar"] = float_prop(other)
        super().__init__(operands=[cond, operand, other_operand], result_types=[result_type], properties=props)


@irdl_op_definition
class GenLayerNormOp(IRDLOperation):
    """nn.LayerNorm over a 1D normalized_shape (hidden_size). weight_name / bias_name
    are set for an elementwise_affine module (bias only if it has one); without them
    there is no affine step."""

    name = "gen.layer_norm"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    hidden_size = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    bias_name = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(
        self,
        operand,
        module_target: str,
        hidden_size: int,
        eps: float,
        result_type,
        weight_name: str | None = None,
        bias_name: str | None = None,
    ):
        props = {
            "module_target": StringAttr(module_target),
            "hidden_size": int_prop(hidden_size),
            "eps": float_prop(eps),
        }
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class GenTransposeOp(IRDLOperation):
    name = "gen.transpose"
    operand = operand_def()
    dim0 = prop_def(IntegerAttr)
    dim1 = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, dim0: int, dim1: int, result_type):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={"dim0": int_prop(dim0), "dim1": int_prop(dim1)},
        )


@irdl_op_definition
class GenPermuteOp(IRDLOperation):
    name = "gen.permute"
    operand = operand_def()
    dims = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, dims, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dims": int_list_prop(dims)})


@irdl_op_definition
class GenViewOp(IRDLOperation):
    name = "gen.view"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class GenNormOp(IRDLOperation):
    """x.norm(p, dim, keepdim); only p = 2 (RMSNorm's) is lowered."""

    name = "gen.norm"
    operand = operand_def()
    p = prop_def(IntegerAttr)
    dim = prop_def(IntegerAttr)
    keepdim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, p: int, dim: int, keepdim: bool, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"p": int_prop(p), "dim": int_prop(dim), "keepdim": bool_prop(keepdim)},
        )


@irdl_op_definition
class GenFlattenOp(IRDLOperation):
    """x.flatten(start_dim, end_dim)."""

    name = "gen.flatten"
    operand = operand_def()
    start_dim = prop_def(IntegerAttr)
    end_dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, start_dim: int, end_dim: int, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"start_dim": int_prop(start_dim), "end_dim": int_prop(end_dim)},
        )


@irdl_op_definition
class GenReshapeOp(IRDLOperation):
    name = "gen.reshape"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class GenReshapeAsOp(IRDLOperation):
    """x.reshape_as(like), rewritten to a gen.reshape to like's shape."""

    name = "gen.reshape_as"
    operand = operand_def()
    like = operand_def()
    result = result_def()

    def __init__(self, operand, like, result_type):
        super().__init__(operands=[operand, like], result_types=[result_type])


@irdl_op_definition
class GenBroadcastOp(IRDLOperation):
    """x.expand(*shape): shape is the call's args, -1s included; the lowering uses the
    result type's shape."""

    name = "gen.broadcast"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class GenUnsqueezeOp(IRDLOperation):
    """x.unsqueeze(dim), a reshape; the lowering uses the result type's shape."""

    name = "gen.unsqueeze"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, dim: int, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class GenSoftmaxOp(IRDLOperation):
    name = "gen.softmax"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, dim: int):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class GenReLUOp(IRDLOperation):
    """nn.ReLU (module_target: the module's path, to rebuild the fx node)."""

    name = "gen.relu"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"module_target": StringAttr(module_target)})


@irdl_op_definition
class GenMaxPool2dOp(IRDLOperation):
    """nn.MaxPool2d; kernel_size / dilation / padding / stride always as 2-tuples."""

    name = "gen.max_pool2d"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    kernel_size = prop_def(ArrayAttr[IntAttr])
    dilation = prop_def(ArrayAttr[IntAttr])
    padding = prop_def(ArrayAttr[IntAttr])
    stride = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, module_target: str, kernel_size, dilation, padding, stride, result_type):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={
                "module_target": StringAttr(module_target),
                "kernel_size": int_list_prop(kernel_size),
                "dilation": int_list_prop(dilation),
                "padding": int_list_prop(padding),
                "stride": int_list_prop(stride),
            },
        )


@irdl_op_definition
class GenAdaptiveAvgPool2dOp(IRDLOperation):
    """nn.AdaptiveAvgPool2d; the lowering supports output_size (1, 1) only and raises
    otherwise."""

    name = "gen.adaptive_avg_pool2d"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    output_size = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, module_target: str, output_size, result_type):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={"module_target": StringAttr(module_target), "output_size": int_list_prop(output_size)},
        )


@irdl_op_definition
class GenGroupLikeNormOp(IRDLOperation):
    """nn.GroupNorm or nn.InstanceNorm{1,2,3}d (num_groups = channels); InstanceNorm
    only with track_running_stats=False."""

    name = "gen.group_like_norm"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    num_groups = prop_def(IntegerAttr)
    channels = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    bias_name = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(
        self, operand, module_target: str, num_groups: int, channels: int, eps: float, result_type,
        weight_name: str | None = None, bias_name: str | None = None,
    ):
        props = {
            "module_target": StringAttr(module_target),
            "num_groups": int_prop(num_groups),
            "channels": int_prop(channels),
            "eps": float_prop(eps),
        }
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class GenLeakyReluOp(IRDLOperation):
    """nn.LeakyReLU."""

    name = "gen.leaky_relu"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    negative_slope = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, negative_slope: float, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"module_target": StringAttr(module_target), "negative_slope": float_prop(negative_slope)},
        )


@irdl_op_definition
class GenSiluModuleOp(IRDLOperation):
    """nn.SiLU as a module, kept apart from gen.silu (F.silu) so it round-trips to a
    call_module node."""

    name = "gen.silu_module"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"module_target": StringAttr(module_target)})


@irdl_op_definition
class GenGeluModuleOp(IRDLOperation):
    """nn.GELU as a module, kept apart from gen.gelu (F.gelu) so it round-trips to a
    call_module node."""

    name = "gen.gelu_module"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"module_target": StringAttr(module_target)})


@irdl_op_definition
class GenUpsampleOp(IRDLOperation):
    """nn.Upsample, mode 'nearest' with integer scale factors (scale_h, scale_w)."""

    name = "gen.upsample"
    operand = operand_def()
    module_target = prop_def(StringAttr)
    mode = prop_def(StringAttr)
    scale_h = prop_def(IntegerAttr)
    scale_w = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, module_target: str, mode: str, scale_h: int, scale_w: int, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={
                "module_target": StringAttr(module_target),
                "mode": StringAttr(mode),
                "scale_h": int_prop(scale_h),
                "scale_w": int_prop(scale_w),
            },
        )


@irdl_op_definition
class GenSliceOp(IRDLOperation):
    """tensor[i0:j0:k0, i1:j1:k1, ...], as common_dialect.CommonSliceOp: starts / stops /
    steps per dim, SLICE_NONE for a None bound, bounds resolved to ints by the bridge."""

    name = "gen.slice"
    operand = operand_def()
    starts = prop_def(ArrayAttr[IntAttr])
    stops = prop_def(ArrayAttr[IntAttr])
    steps = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, starts, stops, steps, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"starts": int_list_prop(starts), "stops": int_list_prop(stops), "steps": int_list_prop(steps)},
        )


Gen = Dialect(
    "gen",
    [
        GenEmbeddingOp,
        GenQuantizePerTensorOp,
        GenDequantizeOp,
        GenQuantizedLinearOp,
        GenQuantizedConv2dOp,
        GenQuantizedMatmulOp,
        GenRowAbsmaxOp,
        GenQuantizePerTokenOp,
        GenDequantizePerTokenOp,
        GenWhereOp,
        GenLayerNormOp,
        GenReLUOp,
        GenMaxPool2dOp,
        GenAdaptiveAvgPool2dOp,
        GenGroupLikeNormOp,
        GenLeakyReluOp,
        GenSiluModuleOp,
        GenGeluModuleOp,
        GenUpsampleOp,
        GenSliceOp,
        GenFlattenOp,
        GenTransposeOp,
        GenPermuteOp,
        GenViewOp,
        GenReshapeOp,
        GenReshapeAsOp,
        GenBroadcastOp,
        GenUnsqueezeOp,
        GenContiguousOp,
        GenAddOp,
        GenMulOp,
        GenDivOp,
        GenSoftmaxOp,
        GenGeluOp,
        GenSiluOp,
        GenEluOp,
        GenSqrtOp,
        GenChunkOp,
        GenCatOp,
        GenMeanOp,
        GenArgmaxOp,
        GenMaxOp,
        GenZerosLikeOp,
        GenScatterOp,
        GenNormOp,
    ],
)
