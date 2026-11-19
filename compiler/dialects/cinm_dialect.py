"""The `cinm` dialect: the generic, hardware-agnostic tensor-algebra
vocabulary the frontend lowers torch ops into, before being lowered
further onto `cim_dialect`'s hardware-realizable ops.
"""

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, FloatAttr, IntAttr, IntegerAttr, StringAttr
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
)

from compiler.dialects.attr_helpers import (
    bool_prop,
    float_prop,
    int_list_prop,
    int_prop,
    is_group_rescaled_raw,
    is_raw_matmul,
)
from compiler.dialects.op_builders import binary_op_either_scalar, binary_op_scalar_rhs, unary_op

CinmVecElMulOp = binary_op_scalar_rhs("cinm.vec_el_mul")
CinmReduceVecOp = unary_op("cinm.reduceVec")
CinmSqrtOp = unary_op("cinm.sqrt")
CinmTruedivOp = binary_op_either_scalar("cinm.truediv")
CinmVecAddOp = binary_op_scalar_rhs("cinm.vec_add")
CinmScalarAddOp = binary_op_scalar_rhs("cinm.scalar_add")
CinmVecDivOp = binary_op_either_scalar("cinm.vec_div")
CinmScalarDivOp = binary_op_either_scalar("cinm.scalar_div")
CinmContiguousOp = unary_op("cinm.contiguous")
CinmGeluOp = unary_op("cinm.gelu")
CinmReluOp = unary_op("cinm.relu")
CinmReciprocalOp = unary_op("cinm.reciprocal")
# 1/sqrt(x), one CMOS circuit (custom_op_library/rsqrt); a layer norm's "rsqrt" impl
# folds reciprocal(sqrt(var + eps)) into it (compiler/ops/layer_norm.py)
CinmRsqrtOp = unary_op("cinm.rsqrt")


@irdl_op_definition
class CinmLUTOp(IRDLOperation):
    """A table lookup at index alpha * index + beta (alpha 1, beta 0 when the index is
    already the table index). op_tag optionally names the lookup (e.g. "dequantize")."""

    name = "cinm.LUTOp"
    table = operand_def()
    index = operand_def()
    alpha = prop_def(FloatAttr)
    beta = prop_def(FloatAttr)
    op_tag = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(self, table, index, result_type, alpha: float, beta: float, op_tag: str | None = None):
        props = {"alpha": float_prop(alpha), "beta": float_prop(beta)}
        if op_tag is not None:
            props["op_tag"] = StringAttr(op_tag)
        super().__init__(operands=[table, index], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmEmbeddingOp(IRDLOperation):
    """table[index, :], one row per index. Not a cinm.LUTOp: an embedding table is a
    large share of the weights (bert-base: ~21%), read once per token, so it is not
    replicated like a function table; it has no cost (a gather, no MAC)."""

    name = "cinm.embedding"
    table = operand_def()
    index = operand_def()
    result = result_def()

    def __init__(self, table, index, result_type):
        super().__init__(operands=[table, index], result_types=[result_type])


@irdl_op_definition
class CinmLUTOpFused(IRDLOperation):
    """A table lookup indexed by several args (no alpha / beta)."""

    name = "cinm.LUTOpFused"
    table = operand_def()
    args = var_operand_def()
    result = result_def()

    def __init__(self, table, args, result_type):
        super().__init__(operands=[table, args], result_types=[result_type])


@irdl_op_definition
class CinmQuantizePerTensorOp(IRDLOperation):
    """1:1 with gen.quantize_per_tensor."""

    name = "cinm.quantize_per_tensor"
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
class CinmDequantizeOp(IRDLOperation):
    """1:1 with gen.dequantize."""

    name = "cinm.dequantize"
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
class CinmQuantizedMatmulOp(IRDLOperation):
    """int8 x int8 matmul, from gen.quantized_linear (after cinm.transpose of the weight,
    with its float32 bias) or gen.quantized_matmul (two activations, no bias).

    Requantizing form: the int32 accumulator times the combined scale (lhs scale x rhs
    scale), plus bias, requantized to output_scale / output_zero_point
    (xdsl_lowering.py's QuantizedMatmulAsCimPattern). Exactly one of combined_scale
    (one float) and combined_scale_name (a const of one scale per output channel).
    Raw form (no scales, no bias): the int32 accumulator alone.
    Group-rescaled raw form (combined_scale_name only, a [groups, out] const): the groups'
    accumulators times their factors, summed, not requantized.
    group_rows: the reduction rows sharing one rhs scale (all of them for one group)."""

    name = "cinm.quantized_matmul"
    lhs = operand_def()
    rhs = operand_def()
    bias = opt_operand_def()
    group_rows = prop_def(IntegerAttr)
    combined_scale = opt_prop_def(FloatAttr)
    combined_scale_name = opt_prop_def(StringAttr)
    output_scale = opt_prop_def(FloatAttr)
    output_zero_point = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, lhs, rhs, group_rows: int, combined_scale: float | None, output_scale: float | None,
                 output_zero_point: int | None, result_type, bias=None, combined_scale_name: str | None = None):
        if group_rows < 1:
            raise ValueError(f"cinm.quantized_matmul: group_rows {group_rows}")
        if is_raw_matmul(combined_scale, combined_scale_name, output_scale, output_zero_point, bias):
            super().__init__(operands=[lhs, rhs, None], result_types=[result_type],
                             properties={"group_rows": int_prop(group_rows)})
            return
        if is_group_rescaled_raw(combined_scale, combined_scale_name, output_scale, output_zero_point, bias):
            if group_rows >= lhs.type.get_shape()[-1]:
                raise ValueError(f"cinm.quantized_matmul: the group-rescaled raw form needs more than one "
                                 f"group of {group_rows} rows")
            super().__init__(operands=[lhs, rhs, None], result_types=[result_type],
                             properties={"group_rows": int_prop(group_rows),
                                         "combined_scale_name": StringAttr(combined_scale_name)})
            return
        if (combined_scale is None) == (combined_scale_name is None):
            raise ValueError("cinm.quantized_matmul: needs exactly one of combined_scale and combined_scale_name")
        if output_scale is None or output_zero_point is None:
            raise ValueError("cinm.quantized_matmul: a requantizing op needs output_scale and output_zero_point "
                             "(or none of the scales and no bias: the raw form)")
        props = {
            "group_rows": int_prop(group_rows),
            "output_scale": float_prop(output_scale),
            "output_zero_point": int_prop(output_zero_point),
        }
        if combined_scale is not None:
            props["combined_scale"] = float_prop(combined_scale)
        else:
            props["combined_scale_name"] = StringAttr(combined_scale_name)
        super().__init__(operands=[lhs, rhs, bias], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmDotProductOp(IRDLOperation):
    name = "cinm.dotProductOp"
    lhs = operand_def()
    rhs = operand_def()
    op_tag = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(self, lhs, rhs, result_type, op_tag: str | None = None):
        props = {}
        if op_tag is not None:
            props["op_tag"] = StringAttr(op_tag)
        super().__init__(operands=[lhs, rhs], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmTransposeOp(IRDLOperation):
    name = "cinm.transpose"
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
class CinmPermuteOp(IRDLOperation):
    name = "cinm.permute"
    operand = operand_def()
    dims = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, dims, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dims": int_list_prop(dims)})


@irdl_op_definition
class CinmReshapeOp(IRDLOperation):
    name = "cinm.reshape"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class CinmViewOp(IRDLOperation):
    name = "cinm.view"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class CinmBroadcastOp(IRDLOperation):
    name = "cinm.broadcast"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class CinmSoftmaxOp(IRDLOperation):
    name = "cinm.softmax"
    operand = operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, dim: int):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class CinmLayerNormOp(IRDLOperation):
    """nn.LayerNorm over the last dim (hidden_size elements), from gen.layer_norm, kept
    whole until compiler/ops/layer_norm.py lowers it by the variant's config.
    weight_name / bias_name name the affine parameters; without them there is none."""

    name = "cinm.layer_norm"
    operand = operand_def()
    hidden_size = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    bias_name = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, hidden_size: int, eps: float, weight_name=None, bias_name=None):
        props = {"hidden_size": int_prop(hidden_size), "eps": float_prop(eps)}
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmRMSNormOp(IRDLOperation):
    """RMSNorm over the last dim (hidden_size elements). A hand-written RMSNorm module
    (e.g. Llama's) reaches gen as primitive ops; xdsl_gen_lowering.py's
    RMSNormAsCinmPattern matches them into this op. weight_name names the scale; no bias."""

    name = "cinm.rms_norm"
    operand = operand_def()
    hidden_size = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, hidden_size: int, eps: float, weight_name=None):
        props = {"hidden_size": int_prop(hidden_size), "eps": float_prop(eps)}
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmGroupLikeNormOp(IRDLOperation):
    """nn.GroupNorm / InstanceNorm, from gen's group-like norm, kept whole until
    compiler/ops/group_norm.py lowers it by the variant's config."""

    name = "cinm.group_like_norm"
    operand = operand_def()
    num_groups = prop_def(IntegerAttr)
    channels = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    bias_name = opt_prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, num_groups: int, channels: int, eps: float, weight_name=None, bias_name=None):
        props = {"num_groups": int_prop(num_groups), "channels": int_prop(channels), "eps": float_prop(eps)}
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CinmSiluOp(IRDLOperation):
    name = "cinm.silu"
    operand = operand_def()
    inplace = opt_prop_def(BoolAttr)
    result = result_def()

    def __init__(self, operand, result_type, inplace: bool = False):
        super().__init__(operands=[operand], result_types=[result_type], properties={"inplace": bool_prop(inplace)})


@irdl_op_definition
class CinmEluOp(IRDLOperation):
    name = "cinm.elu"
    operand = operand_def()
    alpha = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, result_type, alpha: float):
        super().__init__(operands=[operand], result_types=[result_type], properties={"alpha": float_prop(alpha)})


@irdl_op_definition
class CinmLeakyReluOp(IRDLOperation):
    name = "cinm.leaky_relu"
    operand = operand_def()
    negative_slope = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, result_type, negative_slope: float):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"negative_slope": float_prop(negative_slope)},
        )


@irdl_op_definition
class CinmUnfoldOp(IRDLOperation):
    name = "cinm.unfold"
    operand = operand_def()
    kernel_size = prop_def(ArrayAttr[IntAttr])
    dilation = prop_def(ArrayAttr[IntAttr])
    padding = prop_def(ArrayAttr[IntAttr])
    stride = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, kernel_size, result_type, dilation, padding, stride):
        super().__init__(
            operands=[operand],
            result_types=[result_type],
            properties={
                "kernel_size": int_list_prop(kernel_size),
                "dilation": int_list_prop(dilation),
                "padding": int_list_prop(padding),
                "stride": int_list_prop(stride),
            },
        )


@irdl_op_definition
class CinmMaxOp(IRDLOperation):
    """The k largest values along reduce_dim (k = 1: plain max)."""

    name = "cinm.max"
    operand = operand_def()
    reduce_dim = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, reduce_dim: int, k: int):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"reduce_dim": int_prop(reduce_dim), "k": int_prop(k)},
        )


@irdl_op_definition
class CinmArgmaxOp(IRDLOperation):
    """The indices of the k largest values along reduce_dim (cinm.max's indices)."""

    name = "cinm.argmax"
    operand = operand_def()
    reduce_dim = prop_def(IntegerAttr)
    k = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, reduce_dim: int, k: int):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"reduce_dim": int_prop(reduce_dim), "k": int_prop(k)},
        )


CinmNegOp = unary_op("cinm.neg")


@irdl_op_definition
class CinmCatOp(IRDLOperation):
    name = "cinm.cat"
    tensors = var_operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, tensors, result_type, dim: int):
        super().__init__(operands=[tensors], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class CinmScatterOp(IRDLOperation):
    """zeros_like(x).scatter(dim, index, src), as cim_dialect.CimScatterOp."""

    name = "cinm.scatter"
    index = operand_def()
    src = operand_def()
    dim = prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, index, src, result_type, dim: int):
        super().__init__(operands=[index, src], result_types=[result_type], properties={"dim": int_prop(dim)})


# per-token int8 quantization and the select (kernels/gpt2.py), from gen's ops of the
# same names

# max |x| over the last dim: [..., n] -> [..., 1]
CinmRowAbsmaxOp = unary_op("cinm.row_absmax")


@irdl_op_definition
class CinmQuantizePerTokenOp(IRDLOperation):
    """int8 of x with one scale per row, row_max / 127 (gen.quantize_per_token)."""

    name = "cinm.quantize_per_token"
    operand = operand_def()
    row_max = operand_def()
    result = result_def()

    def __init__(self, operand, row_max, result_type):
        super().__init__(operands=[operand, row_max], result_types=[result_type])


@irdl_op_definition
class CinmDequantizePerTokenOp(IRDLOperation):
    """acc * (row_max / 127) * (col_max / 127) (+ bias) (gen.dequantize_per_token):
    exactly one of col_max (an operand) and col_max_scalar; bias optional."""

    name = "cinm.dequantize_per_token"
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
class CinmWhereOp(IRDLOperation):
    """out = cond ? x : other (gen.where): exactly one of other (an operand) and
    other_scalar."""

    name = "cinm.where"
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


Cinm = Dialect(
    "cinm",
    [
        CinmRowAbsmaxOp,
        CinmQuantizePerTokenOp,
        CinmDequantizePerTokenOp,
        CinmWhereOp,
        CinmVecElMulOp,
        CinmReduceVecOp,
        CinmLUTOp,
        CinmEmbeddingOp,
        CinmQuantizePerTensorOp,
        CinmDequantizeOp,
        CinmQuantizedMatmulOp,
        CinmDotProductOp,
        CinmSqrtOp,
        CinmTruedivOp,
        CinmVecAddOp,
        CinmScalarAddOp,
        CinmVecDivOp,
        CinmScalarDivOp,
        CinmTransposeOp,
        CinmPermuteOp,
        CinmReshapeOp,
        CinmViewOp,
        CinmContiguousOp,
        CinmBroadcastOp,
        CinmSoftmaxOp,
        CinmLayerNormOp,
        CinmRMSNormOp,
        CinmGroupLikeNormOp,
        CinmGeluOp,
        CinmSiluOp,
        CinmReluOp,
        CinmRsqrtOp,
        CinmEluOp,
        CinmLeakyReluOp,
        CinmUnfoldOp,
        CinmMaxOp,
        CinmArgmaxOp,
        CinmNegOp,
        CinmCatOp,
        CinmScatterOp,
        CinmReciprocalOp,
        CinmLUTOpFused,
    ],
)
