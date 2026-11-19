"""Python model of rtl/rsqrt_q12_20.v / rtl/rsqrt_q8_8.v (bit-exact) and the fixed-point
helpers the layer-norm stream tests share. Imported by test_layer_norm_stream.py,
test_ailayernorm_stream.py, ../rsqrt/test_rsqrt.py and
run_scripts/measure_accuracy/bert_base_int8_emulator.py."""
import math

IDX_BITS = 5

WIDTH_CFG = {32: dict(frac=20), 16: dict(frac=8)}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def build_seed_table(scale):
    table = {}
    for parity in (0, 1):
        for idx in range(1 << IDX_BITS):
            m = 1.0 + idx / (1 << IDX_BITS)
            f = (1.0 / math.sqrt(m)) if parity == 0 else (1.0 / math.sqrt(2 * m))
            table[(parity, idx)] = round(f * scale)
    return table


SEEDS = {w: build_seed_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def rsqrt_fixed(val, w):
    frac = WIDTH_CFG[w]["frac"]
    scale = 1 << frac
    one_half_q = round(1.5 * scale)
    p = val.bit_length() - 1
    p_real = p - frac
    e = p_real // 2
    parity = p_real & 1
    if p >= IDX_BITS:
        idx = (val >> (p - IDX_BITS)) & ((1 << IDX_BITS) - 1)
    else:
        idx = (val << (IDX_BITS - p)) & ((1 << IDX_BITS) - 1)
    seed = SEEDS[w][(parity, idx)]
    y = (seed >> e) if e >= 0 else (seed << (-e))
    yy_q = (y * y) >> frac
    vyy_q = (val * yy_q) >> frac
    term_q = one_half_q - (vyy_q >> 1)
    y = (y * term_q) >> frac
    return clampW(y, w)
