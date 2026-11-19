"""The integer encoding (sh, m, zp) of a quantize / dequantize circuit for one op's
scale and zero point: the one place it is computed. The compiler encodes from the
scale and zero point on the op in the IR; the variant generator calls the same
functions only to reject a variant whose circuit cannot represent an op's scale.

Circuit impls (custom_op_library/{quantize,dequantize}/rtl): "mant_*" multiply by a
mantissa m and shift right by sh; "pow2_*" only shift (no multiplier); "*_zp" add a
zero point. x is fixed-point at `frac` fraction bits (cim_dialect.FIXED_POINT_FRAC_BITS):
  quantize    q = round(x * m / 2^sh) + zp      m / 2^sh ~ 1 / (2^frac * scale)
  dequantize  x = round((q - zp) * m / 2^sh)    m / 2^sh ~ scale * 2^frac
"""
import math

from xdsl.dialects.builtin import IntegerAttr, StringAttr, i64

from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.cim_dialect import FIXED_POINT_FRAC_BITS
from compiler.ops.dispatch import fx_name_of, require

QUANTIZE = "quantize_per_tensor"
DEQUANTIZE = "dequantize"
CIRCUIT_IMPLS = ("mant_round", "mant_trunc", "mant_round_zp", "pow2_round", "pow2_trunc", "pow2_round_zp")
_MAX_SH = 31
# dequantize's container is a fixed 42 bits; quantize's depends on the widths
_DEQUANTIZE_CONTAINER_BITS = 42


def uses_mantissa(impl: str) -> bool:
    return "pow2" not in impl


def _max_sh(kind: str, impl: str, data_width: int, mantissa_width: int) -> int:
    """Largest shift the circuit allows: the rounding term 2^(sh-1) must fit in its
    intermediate container."""
    if kind == DEQUANTIZE:
        return min(_MAX_SH, _DEQUANTIZE_CONTAINER_BITS - 2)
    container = (1 + data_width + mantissa_width) if uses_mantissa(impl) else data_width
    return min(_MAX_SH, container - 2)


def _target_ratio(kind: str, scale: float, frac: int) -> float:
    """The value m / 2^sh must approximate."""
    return 1.0 / (2.0 ** frac * scale) if kind == QUANTIZE else scale * 2.0 ** frac


def _check_zero_point(fx_name: str, impl: str, zero_point: int) -> None:
    if not impl.endswith("_zp") and zero_point != 0:
        raise ValueError(f"{fx_name}: impl {impl!r} has no zero-point input, but the op's zero_point is "
                         f"{zero_point!r} (no trunc+zp circuit exists)")


def encode_mantissa(kind: str, fx_name: str, scale: float, zero_point: int, impl: str,
                    mantissa_width: int, data_width: int) -> dict:
    """{"sh", "m", "zp" (zp impls)} of a mant_* impl: the largest sh whose rounded m
    fits in mantissa_width bits; raises if none does."""
    if scale <= 0:
        raise ValueError(f"{fx_name}: {kind} scale must be positive, got {scale!r}")
    _check_zero_point(fx_name, impl, zero_point)
    k = _target_ratio(kind, scale, FIXED_POINT_FRAC_BITS[data_width])
    m_max = (1 << mantissa_width) - 1
    for sh in range(_max_sh(kind, impl, data_width, mantissa_width), -1, -1):
        m = round(k * 2 ** sh)
        if 1 <= m <= m_max:
            return {"sh": sh, "m": m, **({"zp": zero_point} if impl.endswith("_zp") else {})}
    raise ValueError(f"{fx_name}: no (m, sh) for {kind} scale={scale!r} at mantissa_width={mantissa_width}, "
                     f"data_width={data_width}")


def encode_pow2(kind: str, fx_name: str, scale: float, zero_point: int, impl: str, data_width: int) -> dict:
    """{"sh", "zp" (zp impls)} of a pow2_* impl: the scale must be a power of two and the
    shift non-negative (these circuits only shift right)."""
    if scale <= 0:
        raise ValueError(f"{fx_name}: {kind} scale must be positive, got {scale!r}")
    _check_zero_point(fx_name, impl, zero_point)
    log2 = math.log2(scale)
    if abs(log2 - round(log2)) >= 1e-9:
        raise ValueError(f"{fx_name}: impl {impl!r} needs a power-of-two scale, got {scale!r}")
    frac = FIXED_POINT_FRAC_BITS[data_width]
    sh = round(frac + log2) if kind == QUANTIZE else round(-frac - log2)
    max_sh = _max_sh(kind, impl, data_width, 0)
    if sh < 0 or sh > max_sh:
        raise ValueError(f"{fx_name}: impl {impl!r} needs sh={sh}, outside [0, {max_sh}]")
    return {"sh": sh, **({"zp": zero_point} if impl.endswith("_zp") else {})}


def stamp_circuit_encoding(kind: str, op, entry: dict) -> None:
    """Handler of a circuit impl: encodes the op's own scale / zero point and stamps
    variant, sh, m / mantissa_width (mant impls), zp (zp impls) and accel_data_width
    onto the unchanged op, for the cim lowering."""
    fx_name = fx_name_of(op)
    impl = require(entry, "impl", fx_name)
    data_width = require(entry, "data_width", fx_name)
    scale, zero_point = float_from_prop(op.scale), int_from_prop(op.zero_point)
    if uses_mantissa(impl):
        mantissa_width = require(entry, "mantissa_width", fx_name)
        encoding = encode_mantissa(kind, fx_name, scale, zero_point, impl, mantissa_width, data_width)
        op.attributes["m"] = IntegerAttr(encoding["m"], i64)
        op.attributes["mantissa_width"] = IntegerAttr(int(mantissa_width), i64)
    else:
        encoding = encode_pow2(kind, fx_name, scale, zero_point, impl, data_width)
    op.attributes["variant"] = StringAttr(impl)
    op.attributes["sh"] = IntegerAttr(encoding["sh"], i64)
    op.attributes["accel_data_width"] = IntegerAttr(int(data_width), i64)
    if "zp" in encoding:
        op.attributes["zp"] = IntegerAttr(encoding["zp"], i64)
