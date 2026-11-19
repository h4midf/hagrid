"""The `cim` dialect: the hardware-realizable vocabulary `cinm_dialect`
ops get lowered into.
"""

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, FloatAttr, IntAttr, IntegerAttr, StringAttr
from xdsl.ir import Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
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
)
from compiler.dialects.op_builders import binary_op, binary_op_scalar_rhs, unary_op


@irdl_attr_definition
class FixedType(ParametrizedAttribute, TypeAttribute):
    """Signed fixed-point number: `width` bits, `frac` of them after the binary point.
    The hardware format of every non-integer value."""

    name = "cim.fixed"

    width: IntAttr
    frac: IntAttr

    def __init__(self, width: int, frac: int):
        super().__init__(IntAttr(width), IntAttr(frac))


# a variant's data_width -> fraction bits: 32 = Q12.20, 16 = Q8.8 (custom_op_library's formats)
FIXED_POINT_FRAC_BITS = {32: 20, 16: 8}


def fixed_type_for(data_width: int) -> FixedType:
    """The fixed-point format of a variant's data_width."""
    return FixedType(data_width, FIXED_POINT_FRAC_BITS[data_width])

CimVecAddOp = binary_op_scalar_rhs("cim.vec_add")
CimVecElMulOp = binary_op_scalar_rhs("cim.vec_el_mul")
CimReluOp = unary_op("cim.relu")
CimSiluOp = unary_op("cim.silu")
# 1/sqrt(x), one CMOS circuit (custom_op_library/rsqrt)
CimRsqrtOp = unary_op("cim.rsqrt")
CimContiguousOp = unary_op("cim.contiguous")
CimLutSearchVectorOp = binary_op("cim.lut_search_vector")
# table[index, :] (lhs = table, rhs = index); its own op, costed apart from
# lut_search_vector (cinm_dialect.CinmEmbeddingOp)
CimEmbeddingOp = binary_op("cim.embedding")

# per-token int8 quantization (kernels/gpt2.py); per-token dequantization lowers to
# cim.vec_el_mul / cim.vec_add

# max |x| over the last dim, [..., n] -> [..., 1]: custom_op_library/token_quant's
# absmax_reduce_tree
CimRowAbsmaxOp = unary_op("cim.row_absmax")


@irdl_op_definition
class CimQuantizePerTokenOp(IRDLOperation):
    """int8 of x with one scale per row, row_max / 127: custom_op_library/token_quant's
    token_scale (one per row: the m, sh of 127 / row_max) and quantize's
    quant_mant_round (one per element). x [..., n], row_max [..., 1] -> int8 [..., n]."""

    name = "cim.quantize_per_token"
    operand = operand_def()
    row_max = operand_def()
    result = result_def()

    def __init__(self, operand, row_max, result_type):
        super().__init__(operands=[operand, row_max], result_types=[result_type])


@irdl_op_definition
class CimVecSelectOp(IRDLOperation):
    """out = cond ? x : fill, elementwise: custom_op_library/vec_select (a 2-to-1
    multiplexer per element). cond a bool tensor (e.g. GPT-2's stored causal mask), fill
    one word shared by the elements, as the circuit takes it."""

    name = "cim.vec_select"
    cond = operand_def()
    operand = operand_def()
    fill = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, cond, operand, fill: float, result_type):
        super().__init__(operands=[cond, operand], result_types=[result_type], properties={"fill": float_prop(fill)})


@irdl_op_definition
class CimEluOp(IRDLOperation):
    """elu(x) on the circuit (custom_op_library/elu), one circuit per alpha: the timing
    model picks the characterized row by alpha."""

    name = "cim.elu"
    operand = operand_def()
    alpha = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, result_type, alpha: float):
        super().__init__(operands=[operand], result_types=[result_type], properties={"alpha": float_prop(alpha)})


@irdl_op_definition
class CimLeakyReluOp(IRDLOperation):
    """leaky_relu(x) on the circuit (custom_op_library/leaky_relu), one circuit per
    negative_slope: the timing model picks the characterized row by it."""

    name = "cim.leaky_relu"
    operand = operand_def()
    negative_slope = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, operand, result_type, negative_slope: float):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"negative_slope": float_prop(negative_slope)},
        )


@irdl_op_definition
class CimGeluOp(IRDLOperation):
    """gelu(x) on the circuit `variant` (compiler/ops/gelu.py): the timing model picks
    the characterized row by it."""

    name = "cim.gelu"
    operand = operand_def()
    variant = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, variant: str):
        super().__init__(operands=[operand], result_types=[result_type], properties={"variant": StringAttr(variant)})


@irdl_op_definition
class CimLayerNormOp(IRDLOperation):
    """layer_norm over the last dim (hidden_size elements) on the circuit `variant`
    (compiler/ops/layer_norm.py): the timing model picks the characterized row by variant
    and hidden_size. weight_name / bias_name name the affine parameters."""

    name = "cim.layer_norm"
    operand = operand_def()
    hidden_size = prop_def(IntegerAttr)
    eps = prop_def(FloatAttr)
    weight_name = opt_prop_def(StringAttr)
    bias_name = opt_prop_def(StringAttr)
    variant = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, hidden_size: int, eps: float, variant: str,
                 weight_name=None, bias_name=None):
        props = {"hidden_size": int_prop(hidden_size), "eps": float_prop(eps), "variant": StringAttr(variant)}
        if weight_name is not None:
            props["weight_name"] = StringAttr(weight_name)
        if bias_name is not None:
            props["bias_name"] = StringAttr(bias_name)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CimRMSNormOp(IRDLOperation):
    """rms_norm over the last dim (hidden_size elements) on the circuit
    (custom_op_library/rms_norm): the timing model picks the characterized row by
    hidden_size. weight_name names the scale; RMSNorm has no bias."""

    name = "cim.rms_norm"
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
class CimQuantizeOp(IRDLOperation):
    """Quantize to int8 on the circuit `variant` (compiler/ops/quantize_per_tensor.py).
    sh / m / zp encode this op's scale and zero point (compiler/ops/quant_encoding.py);
    mantissa_width is the width the circuit is characterized at, which m fits in. m, zp
    and mantissa_width are present only for the variants with those ports. The timing
    model picks the characterized row by variant and mantissa_width."""

    name = "cim.quantize"
    operand = operand_def()
    # a SmoothQuant x * (1 / s) folded into this quantize (compiler/xdsl_fold_smoothing.py):
    # one (m, sh) per channel of the last dim, used instead of the op's m / sh
    channel_table = opt_operand_def()
    variant = prop_def(StringAttr)
    sh = prop_def(IntegerAttr)
    m = opt_prop_def(IntegerAttr)
    zp = opt_prop_def(IntegerAttr)
    mantissa_width = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, variant: str, sh: int, m=None, zp=None, mantissa_width=None,
                 channel_table=None):
        props = {"variant": StringAttr(variant), "sh": int_prop(sh)}
        if m is not None:
            props["m"] = int_prop(m)
        if zp is not None:
            props["zp"] = int_prop(zp)
        if mantissa_width is not None:
            props["mantissa_width"] = int_prop(mantissa_width)
        super().__init__(operands=[operand, channel_table], result_types=[result_type], properties=props)


@irdl_op_definition
class CimDequantizeOp(IRDLOperation):
    """Dequantize from int8, the reverse of CimQuantizeOp, with the same properties."""

    name = "cim.dequantize"
    operand = operand_def()
    variant = prop_def(StringAttr)
    sh = prop_def(IntegerAttr)
    m = opt_prop_def(IntegerAttr)
    zp = opt_prop_def(IntegerAttr)
    mantissa_width = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, variant: str, sh: int, m=None, zp=None, mantissa_width=None):
        props = {"variant": StringAttr(variant), "sh": int_prop(sh)}
        if m is not None:
            props["m"] = int_prop(m)
        if zp is not None:
            props["zp"] = int_prop(zp)
        if mantissa_width is not None:
            props["mantissa_width"] = int_prop(mantissa_width)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CimMatvecOp(IRDLOperation):
    """lhs x rhs on the arrays: a float matvec, or the int8 x int8 product of a
    quantized matmul before its requantize (i8 in, i32 out). out_shape is the tiled
    result shape the mapping expects (e.g. [rounds, tile]); untiled matvecs have none."""

    name = "cim.MatvecOp"
    lhs = operand_def()
    rhs = operand_def()
    out_shape = opt_prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, lhs, rhs, result_type, out_shape=None):
        props = {}
        if out_shape is not None:
            props["out_shape"] = int_list_prop(out_shape)
        super().__init__(operands=[lhs, rhs], result_types=[result_type], properties=props)


@irdl_op_definition
class CimSoftmaxOp(IRDLOperation):
    """softmax along dim on the circuit `variant` (compiler/ops/softmax.py): the timing
    model picks the characterized row by it."""

    name = "cim.softmax"
    operand = operand_def()
    dim = opt_prop_def(IntegerAttr)
    variant = prop_def(StringAttr)
    result = result_def()

    def __init__(self, operand, result_type, variant: str, dim: int = -1):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"dim": int_prop(dim), "variant": StringAttr(variant)},
        )


@irdl_op_definition
class CimLutSearchOp(IRDLOperation):
    """table[round(index * alpha + beta)], a 1D-table lookup."""

    name = "cim.lut_search"
    table = operand_def()
    index = operand_def()
    alpha = prop_def(FloatAttr)
    beta = prop_def(FloatAttr)
    result = result_def()

    def __init__(self, table, index, alpha: float, beta: float, result_type):
        super().__init__(
            operands=[table, index],
            result_types=[result_type],
            properties={"alpha": float_prop(alpha), "beta": float_prop(beta)},
        )


@irdl_op_definition
class CimReduceVecOp(IRDLOperation):
    """Sum along reduce_dim; out_shape is the tiled result shape, if tiled."""

    name = "cim.reduce_vec"
    operand = operand_def()
    reduce_dim = prop_def(IntegerAttr)
    out_shape = opt_prop_def(ArrayAttr[IntAttr])
    tiles_dim = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, reduce_dim: int, result_type, out_shape=None, tiles_dim=None):
        props = {"reduce_dim": int_prop(reduce_dim)}
        if out_shape is not None:
            props["out_shape"] = int_list_prop(out_shape)
        if tiles_dim is not None:
            props["tiles_dim"] = int_prop(tiles_dim)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class CimArgmaxOp(IRDLOperation):
    """Indices of the k largest values along reduce_dim (k = 1: argmax; k > 1: top-k,
    e.g. MoE routing). The index output of the comparator tree CimMaxOp shares."""

    name = "cim.argmax"
    operand = operand_def()
    reduce_dim = opt_prop_def(IntegerAttr)
    k = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, reduce_dim: int = -1, k: int = 1):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"reduce_dim": int_prop(reduce_dim), "k": int_prop(k)},
        )


@irdl_op_definition
class CimMaxOp(IRDLOperation):
    """The k largest values along reduce_dim: the value output of CimArgmaxOp's
    comparator tree."""

    name = "cim.max"
    operand = operand_def()
    reduce_dim = opt_prop_def(IntegerAttr)
    k = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, operand, result_type, reduce_dim: int = -1, k: int = 1):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"reduce_dim": int_prop(reduce_dim), "k": int_prop(k)},
        )


@irdl_op_definition
class CimTransposeOp(IRDLOperation):
    name = "cim.transpose"
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
class CimPermuteOp(IRDLOperation):
    name = "cim.permute"
    operand = operand_def()
    dims = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, dims, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"dims": int_list_prop(dims)})


@irdl_op_definition
class CimViewOp(IRDLOperation):
    name = "cim.view"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class CimReshapeOp(IRDLOperation):
    name = "cim.reshape"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


@irdl_op_definition
class CimBroadcastOp(IRDLOperation):
    name = "cim.broadcast"
    operand = operand_def()
    out_shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, out_shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"out_shape": int_list_prop(out_shape)})


@irdl_op_definition
class CimCatOp(IRDLOperation):
    name = "cim.cat"
    tensors = var_operand_def()
    dim = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, tensors, result_type, dim: int = 0):
        super().__init__(operands=[tensors], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class CimScatterOp(IRDLOperation):
    """zeros_like(x).scatter(dim, index, src): src written into an all-zero buffer at
    index along dim, every other position zero (fused from gen.zeros_like + gen.scatter
    in xdsl_gen_lowering.py's ScatterAsCinmPattern; a non-zero base is not supported).
    Costed as one addressed write per element."""

    name = "cim.scatter"
    index = operand_def()
    src = operand_def()
    dim = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(self, index, src, result_type, dim: int = -1):
        super().__init__(operands=[index, src], result_types=[result_type], properties={"dim": int_prop(dim)})


@irdl_op_definition
class CimUnfoldOp(IRDLOperation):
    name = "cim.unfold"
    operand = operand_def()
    kernel_size = prop_def(ArrayAttr[IntAttr])
    dilation = opt_prop_def(ArrayAttr[IntAttr])
    padding = opt_prop_def(ArrayAttr[IntAttr])
    stride = opt_prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, kernel_size, result_type, dilation=1, padding=0, stride=1):
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
class CimSaveStationaryOp(IRDLOperation):
    """Marks a value as stored in the stationary (weight) arrays."""

    name = "cim.save_stationary"
    operand = operand_def()
    transposed = opt_prop_def(BoolAttr)
    out_shape = opt_prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, result_type, transposed: bool = False, out_shape=None):
        props = {"transposed": bool_prop(transposed)}
        if out_shape is not None:
            props["out_shape"] = int_list_prop(out_shape)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


Cim = Dialect(
    "cim",
    [
        CimMatvecOp,
        CimRowAbsmaxOp,
        CimQuantizePerTokenOp,
        CimVecSelectOp,
        CimVecAddOp,
        CimVecElMulOp,
        CimGeluOp,
        CimReluOp,
        CimRsqrtOp,
        CimSiluOp,
        CimEluOp,
        CimLeakyReluOp,
        CimLayerNormOp,
        CimRMSNormOp,
        CimQuantizeOp,
        CimDequantizeOp,
        CimSoftmaxOp,
        CimLutSearchOp,
        CimLutSearchVectorOp,
        CimEmbeddingOp,
        CimReduceVecOp,
        CimArgmaxOp,
        CimMaxOp,
        CimTransposeOp,
        CimPermuteOp,
        CimViewOp,
        CimContiguousOp,
        CimBroadcastOp,
        CimCatOp,
        CimScatterOp,
        CimUnfoldOp,
        CimSaveStationaryOp,
        CimReshapeOp,
    ],
    [FixedType],
)
