"""Error of rms_norm's fixed-point scheme against true RMSNorm, at both widths (pure
Python, no RTL). Uses the same rsqrt seed table and one Newton iteration as
layer_norm/approximation_error.py.
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


def true_rms_norm(xs, gamma=1.0, eps=1e-6):
    n = len(xs)
    ms = sum(x * x for x in xs) / n
    inv = 1.0 / (math.sqrt(ms) + eps)
    return [(x * inv) * gamma for x in xs]


def fixed_point_rms_norm(xs_float, w, gamma=1.0, eps=1e-6):
    frac = WIDTH_CFG[w]["frac"]
    scale = 1 << frac
    n = len(xs_float)
    levels = int(math.log2(n))
    half = 1 << (frac - 1)
    xs = [round(x * scale) for x in xs_float]

    sqs = [clampW((x * x + half) >> frac, w) for x in xs]
    ms = clampW(sum(sqs) >> levels, w)

    if ms == 0:
        s = 0
    else:
        r1 = rsqrt_fixed(ms, w)
        s = clampW((ms * r1 + half) >> frac, w)

    eps_q = round(eps * scale)
    s_eps = clampW(s + eps_q, w)
    v2 = clampW((s_eps * s_eps + half) >> frac, w)
    inv = rsqrt_fixed(v2, w) if v2 > 0 else 0

    norms = [clampW((x * inv + half) >> frac, w) for x in xs]
    gamma_q = round(gamma * scale)
    scaled = [clampW((nv * gamma_q + half) >> frac, w) for nv in norms]
    return [s_ / scale for s_ in scaled]


def run_for_width(w, rng):
    max_abs_err = 0.0
    sum_abs_err = 0.0
    n = 0
    for _ in range(5000):
        xs = [rng.uniform(-10, 10) for _ in range(16)]
        got = fixed_point_rms_norm(xs, w)
        exp = true_rms_norm(xs)
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
        got = fixed_point_rms_norm(xs, w)
        exp = true_rms_norm(xs)
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
