"""requantize: the int8 requantize at the end of every int8 matmul (quantized Linear,
conv, quantized.matmul), one function per impl.

The matmul's config entry (kind "linear" / "matmul") names it as requantize_impl, the
quantize impl of the variant:
- "lut": nothing here; xdsl_lowering.QuantizedMatmulAsCimPattern looks the result up
  in the shared 256-entry identity table (the same lookup as quantize "lut").
- a circuit impl (quant_encoding.CIRCUIT_IMPLS, custom_op_library/quantize): the
  (sh, m, zp) encoding of the matmul's
  output scale / zero point is stamped onto the cinm.quantized_matmul as
  requant_variant / requant_sh / requant_m / requant_mantissa_width / requant_zp;
  QuantizedMatmulAsCimPattern then ends the matmul in cim.quantize instead of the LUT.
A matmul in the raw form (no output scale: the i32 accumulator is its result) has no
requantize.
"""
from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.ops.dispatch import fx_name_of, require
from compiler.ops.quant_encoding import CIRCUIT_IMPLS, QUANTIZE, encode_mantissa, encode_pow2, uses_mantissa

_KINDS = ("linear", "matmul")


def stamp_requantize(op: nd.CinmQuantizedMatmulOp, entry: dict) -> None:
    """Stamps the requantize circuit's encoding onto `op` (a circuit requantize_impl)."""
    fx_name = fx_name_of(op)
    impl = require(entry, "requantize_impl", fx_name)
    data_width = require(entry, "data_width", fx_name)
    scale, zero_point = float_from_prop(op.output_scale), int_from_prop(op.output_zero_point)
    if uses_mantissa(impl):
        mantissa_width = require(entry, "requantize_mantissa_width", fx_name)
        encoding = encode_mantissa(QUANTIZE, fx_name, scale, zero_point, impl, mantissa_width, data_width)
        op.attributes["requant_m"] = IntegerAttr(encoding["m"], i64)
        op.attributes["requant_mantissa_width"] = IntegerAttr(int(mantissa_width), i64)
    else:
        encoding = encode_pow2(QUANTIZE, fx_name, scale, zero_point, impl, data_width)
    op.attributes["requant_variant"] = StringAttr(impl)
    op.attributes["requant_sh"] = IntegerAttr(encoding["sh"], i64)
    op.attributes["requant_data_width"] = IntegerAttr(int(data_width), i64)
    if "zp" in encoding:
        op.attributes["requant_zp"] = IntegerAttr(encoding["zp"], i64)


class RequantizeCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmQuantizedMatmulOp, rewriter: PatternRewriter) -> None:
        if op.output_scale is None or "requant_variant" in op.attributes:
            return   # raw form (no requantize), or already stamped
        fx_name = fx_name_of(op)
        entry = self.config.get(fx_name)
        if entry is None or entry.get("kind") not in _KINDS:
            raise KeyError(f"{fx_name}: no linear / matmul entry in the application config for its requantize")
        impl = require(entry, "requantize_impl", fx_name)
        if impl == "lut":
            return
        if impl not in CIRCUIT_IMPLS:
            raise ValueError(f"{fx_name}: requantize_impl {impl!r} is neither 'lut' nor one of {CIRCUIT_IMPLS}")
        stamp_requantize(op, entry)


def run_requantize_canon(module, config: dict) -> None:
    PatternRewriteWalker(RequantizeCanonPattern(config)).rewrite_module(module)
