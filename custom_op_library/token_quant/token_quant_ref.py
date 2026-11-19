"""Python reference of per-token (dynamic, one scale per row) int8 quantisation, bit-exact
with rtl/absmax_reduce_tree.v, rtl/token_scale.v and ../quantize/rtl/quant_mant_round.v.

All values are the integers of a fixed-point format (Q12.20 at 32 bits): x and the row
max M share it, so the factor 127 / M does not depend on it."""

MW = 12
SH_MAX = 31


def absmax(row: list) -> int:
    """max |x| over the row (absmax_reduce_tree)."""
    return max(abs(x) for x in row)


def token_scale(m_abs: int, mw: int = MW, sh_max: int = SH_MAX) -> tuple:
    """(m, sh) with m * 2^-sh ~= 127 / m_abs (token_scale)."""
    if m_abs == 0:
        return 0, 0
    sh = min(m_abs.bit_length() + mw - 8, sh_max)
    m = ((127 << sh) + (m_abs >> 1)) // m_abs
    return m, sh


def quant_mant_round(x: int, m: int, sh: int) -> int:
    """clamp_int8(round(x * m * 2^-sh)), round to nearest, ties up (quant_mant_round)."""
    half = (1 << (sh - 1)) if sh > 0 else 0
    return max(-128, min(127, (x * m + half) >> sh))


def quantize_row(row: list) -> tuple:
    """(int8 row, row max M): the per-token quantisation of one row."""
    m_abs = absmax(row)
    m, sh = token_scale(m_abs)
    return [quant_mant_round(x, m, sh) for x in row], m_abs
