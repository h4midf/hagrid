"""Error of the fixed-point rsqrt-based layer norm against true layer normalisation,
at both widths (pure Python, no RTL). Also the derivation of the rsqrt seed-table size
(5 bits, 32 entries per parity) and Newton iteration count (1) in
rtl/rsqrt_q12_20.v / rsqrt_q8_8.v:
  W=32/Q12.20: a bare seed gives 1.5% max relative error over val in [1e-6, 4000],
    one iteration 0.08%, a second iteration no better.
  W=16/Q8.8: one iteration gives 7.8% over the LayerNorm variance range [0.01, 20],
    a second 6.3% (limited by the 5-bit seed table and Q8.8 rounding).
"""
import math
import random

IDX_BITS = 5
WIDTH_CFG = {32: dict(frac=20), 16: dict(frac=8)}


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


def rsqrt_fixed(val, w, iters=1):
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
    y = SEEDS[w][(parity, idx)]
    y = (y >> e) if e >= 0 else (y << (-e))
    for _ in range(iters):
        yy_q = (y * y) >> frac
        vyy_q = (val * yy_q) >> frac
        term_q = one_half_q - (vyy_q >> 1)
        y = (y * term_q) >> frac
    return clampW(y, w)


def true_layer_norm(xs, gamma=1.0, beta=0.0, eps=1e-5):
    n = len(xs)
    mean = sum(xs) / n
    var = sum((x - mean) ** 2 for x in xs) / n
    inv_std = 1.0 / math.sqrt(var + eps)
    return [((x - mean) * inv_std) * gamma + beta for x in xs]


def fixed_point_layer_norm(xs_float, w, gamma=1.0, beta=0.0, eps=1e-5):
    frac = WIDTH_CFG[w]["frac"]
    scale = 1 << frac
    n = len(xs_float)
    levels = int(math.log2(n))
    half = 1 << (frac - 1)
    xs = [round(x * scale) for x in xs_float]
    mean = clampW(sum(xs) >> levels, w)
    diffs = [clampW(x - mean, w) for x in xs]
    sqs = [clampW((d * d + half) >> frac, w) for d in diffs]
    var = clampW(sum(sqs) >> levels, w)
    var_eps = clampW(var + round(eps * scale), w)
    inv_std = rsqrt_fixed(var_eps, w) if var_eps > 0 else 0
    norms = [clampW((d * inv_std + half) >> frac, w) for d in diffs]
    gamma_q, beta_q = round(gamma * scale), round(beta * scale)
    scaled = [clampW((nv * gamma_q + half) >> frac, w) for nv in norms]
    return [clampW(s + beta_q, w) / scale for s in scaled]


def run_for_width(w, rng):
    max_abs_err = 0.0
    sum_abs_err = 0.0
    n = 0
    for _ in range(5000):
        xs = [rng.uniform(-10, 10) for _ in range(16)]
        got = fixed_point_layer_norm(xs, w)
        exp = true_layer_norm(xs)
        for g, e in zip(got, exp):
            err = abs(g - e)
            sum_abs_err += err
            n += 1
            max_abs_err = max(max_abs_err, err)
    print(f"W={w}: n={n} mean_abs_err={sum_abs_err / n:.6f} max_abs_err={max_abs_err:.6f}")

    max_abs_err_c = 0.0
    sum_abs_err_c = 0.0
    n2 = 0
    for _ in range(5000):
        base = rng.uniform(-5, 5)
        xs = [base + rng.uniform(-1, 1) for _ in range(16)]
        got = fixed_point_layer_norm(xs, w)
        exp = true_layer_norm(xs)
        for g, e in zip(got, exp):
            err = abs(g - e)
            sum_abs_err_c += err
            n2 += 1
            max_abs_err_c = max(max_abs_err_c, err)
    print(f"  clustered (spread +-1 around a random base): "
          f"mean_abs_err={sum_abs_err_c / n2:.6f} max_abs_err={max_abs_err_c:.6f}")


def main():
    rng = random.Random(89)
    for w in (32, 16):
        run_for_width(w, rng)


if __name__ == "__main__":
    main()
