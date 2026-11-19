"""Bit-exact reference of the SFU's FP16 units (rtl/*.v), for the tests.

Format (our choice; HyFlexPIM Sec. 5.1 says only 'FP16'): IEEE binary16, 1 sign,
5 exponent (bias 15), 10 fraction bits.
  - Subnormal inputs are read as zero; results below the smallest normal (2^-14),
    after rounding, are flushed to +0. Every zero result is +0 (0x0000).
  - Results above the largest normal saturate to +-65504 (0x7BFF / 0xFBFF).
    Exponent 31 is read as an ordinary exponent (no Inf / NaN).
  - Rounding: to nearest, ties to even.
add / sub / mul / div / sqrt are the exact result rounded once (computed with
Fractions); exp and to_int8 follow the RTL's algorithm step by step.
"""
from fractions import Fraction
import math

MAX_FINITE = 0x7BFF


def decode(h):
    """Exact value of an FP16 bit pattern (subnormals read as zero)."""
    s, e, m = (h >> 15) & 1, (h >> 10) & 0x1F, h & 0x3FF
    if e == 0:
        return Fraction(0)
    v = Fraction(1024 + m, 1024) * Fraction(2) ** (e - 15)
    return -v if s else v


def round_half_even(x: Fraction) -> int:
    q, r = divmod(x.numerator, x.denominator)
    twice = 2 * r
    if twice > x.denominator or (twice == x.denominator and q & 1):
        q += 1
    return q


def encode(v: Fraction) -> int:
    """v rounded to FP16 (nearest even), then flushed / saturated as above."""
    if v == 0:
        return 0
    s = 1 if v < 0 else 0
    a = -v if s else v
    e = a.numerator.bit_length() - a.denominator.bit_length()
    if Fraction(2) ** e > a:
        e -= 1
    q = round_half_even(a * Fraction(2) ** (10 - e))  # 1024 <= q <= 2048
    if q == 2048:
        q, e = 1024, e + 1
    biased = e + 15
    if biased >= 31:
        return (s << 15) | MAX_FINITE
    if biased <= 0:
        return 0
    return (s << 15) | (biased << 10) | (q - 1024)


def add(a, b):
    return encode(decode(a) + decode(b))


def sub(a, b):
    return encode(decode(a) - decode(b))


def mul(a, b):
    return encode(decode(a) * decode(b))


def div(a, b):
    """Division by zero saturates to +-65504 (sign of a XOR sign of b, as IEEE gives
    +-Inf; +65504 for 0 / 0)."""
    va, vb = decode(a), decode(b)
    if vb == 0:
        return (((a ^ b) >> 15) & 1 if va != 0 else 0) << 15 | MAX_FINITE
    return encode(va / vb)


def sqrt(a):
    """sqrt of a negative number or zero is +0."""
    v = decode(a)
    if v <= 0:
        return 0
    # exact rounding: v = M * 2^k with M an integer; sqrt via isqrt with enough bits
    e = (a >> 10) & 0x1F
    M = 1024 + (a & 0x3FF)          # v = M * 2^(e - 25)
    k = e - 25
    if k % 2:
        M, k = M * 2, k - 1
    # sqrt(v) = sqrt(M * 2^(2F)) * 2^(k/2 - F), F large enough to decide the rounding
    F = 40
    r = math.isqrt(M << (2 * F))
    exact = r * r == (M << (2 * F))
    val = Fraction(r) * Fraction(2) ** (k // 2 - F)
    if not exact:  # r is truncated: nudge below the next representable to keep the sticky bit
        val += Fraction(1, 2) * Fraction(2) ** (k // 2 - F)
    return encode(val)


def fmax(a, b):
    """The larger value; inputs flushed first; ties give a."""
    fa = 0 if (a >> 10) & 0x1F == 0 else a
    fb = 0 if (b >> 10) & 0x1F == 0 else b
    return fb if decode(fb) > decode(fa) else fa


def to_int8(a):
    """FP16 -> int8, nearest even, saturating to [-128, 127]."""
    q = round_half_even(abs(decode(a)))
    v = -q if (a >> 15) & 1 else q
    return max(-128, min(127, v))


def int8_to_fp16(n):
    """int8 (two's complement value -128..127) -> FP16, exact."""
    return encode(Fraction(n))


def ldexp(a, n):
    """a * 2^n for an int8 n: exact apart from flush / saturation."""
    return encode(decode(a) * Fraction(2) ** n)


# ---- exp, computed on the lane's own units (rtl/sfu_lane.v), as in the HyFlexPIM
# authors' prior work: TransPIM computes exp by a five-order Taylor series with
# multiplications and additions, PIM-GPT computes exp with Taylor terms on shared FP
# adders and multipliers. Every step is one of the units above, in FP16:
#   t = x log2(e) (mul), n = round(t) (to_int8), f = t - n (int8_to_fp16, sub),
#   r = f ln2 (mul), e^r = sum_{k<=5} r^k / k! by Horner (5 x mul + add),
#   e^x = 2^n e^r (ldexp).
# Exact (correctly rounded) on 93% of FP16 inputs, within 1 ulp on 99.6%; the rest,
# up to 10 ulp, at large |x|, where t = x log2(e) is rounded to FP16.
EXP_LOG2E = encode(Fraction(math.log2(math.e)))
EXP_LN2 = encode(Fraction(math.log(2)))
EXP_TAYLOR = [encode(Fraction(1, math.factorial(k))) for k in range(6)]


def exp(a):
    t = mul(a, EXP_LOG2E)
    n = to_int8(t)
    f = sub(t, int8_to_fp16(n))
    r = mul(f, EXP_LN2)
    p = EXP_TAYLOR[5]
    for k in (4, 3, 2, 1, 0):
        p = add(mul(p, r), EXP_TAYLOR[k])
    return ldexp(p, n)
