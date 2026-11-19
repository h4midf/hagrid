"""Error of gelu.v's fixed-point 256-segment piecewise-linear scheme against true GELU,
at both widths. Pure Python, no RTL (test_gelu.py checks the RTL). It chose the segment
count (128 missed the max-error bar) and the tables in rtl/gelu.v.
"""
import math
import random

N_SEG = 256
DOMAIN_HALF = 8
WIDTH_CFG = {32: dict(frac=20, shift=16), 16: dict(frac=8, shift=4)}


def true_gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def build_table(scale):
    seg_width = (2 * DOMAIN_HALF) / N_SEG
    offs, slopes = [], []
    for k in range(N_SEG):
        x_lo = -DOMAIN_HALF + k * seg_width
        x_hi = x_lo + seg_width
        y_lo, y_hi = true_gelu(x_lo), true_gelu(x_hi)
        offs.append(round(y_lo * scale))
        slopes.append(round((y_hi - y_lo) * scale))
    return offs, slopes


TABLES = {w: build_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def fixed_point_gelu(x_float, w):
    cfg = WIDTH_CFG[w]
    scale = 1 << cfg["frac"]
    dom_q = DOMAIN_HALF * scale
    offs, slopes = TABLES[w]
    xq = round(x_float * scale)
    if xq < -dom_q:
        return 0.0
    if xq >= dom_q:
        return xq / scale
    s = xq + dom_q
    idx = s >> cfg["shift"]
    intra = s & ((1 << cfg["shift"]) - 1)
    off, slope = offs[idx], slopes[idx]
    return (off + ((slope * intra) >> cfg["shift"])) / scale


def run_for_width(w, rng):
    max_abs_err = 0.0
    sum_abs_err = 0.0
    n = 0
    worst = None
    for _ in range(50000):
        x = rng.uniform(-10, 10)
        got = fixed_point_gelu(x, w)
        exp = true_gelu(x) if abs(x) < DOMAIN_HALF else (0.0 if x < 0 else x)
        err = abs(got - exp)
        sum_abs_err += err
        n += 1
        if err > max_abs_err:
            max_abs_err = err
            worst = (x, got, exp)
    print(f"W={w}: n={n} mean_abs_err={sum_abs_err / n:.7f} max_abs_err={max_abs_err:.7f}")
    print("  worst case x =", worst[0], " fixed-point:", round(worst[1], 5), " true GELU:", round(worst[2], 5))

    max_abs_err_c = 0.0
    sum_abs_err_c = 0.0
    n2 = 0
    for _ in range(50000):
        x = rng.uniform(-4, 4)
        got = fixed_point_gelu(x, w)
        exp = true_gelu(x)
        err = abs(got - exp)
        sum_abs_err_c += err
        n2 += 1
        max_abs_err_c = max(max_abs_err_c, err)
    print(f"  clustered (x in [-4,4]): mean_abs_err={sum_abs_err_c / n2:.7f} max_abs_err={max_abs_err_c:.7f}")


IBERT_CONSTS = {
    32: dict(frac=20, one=1048576, inv_sqrt2=741455, neg_b=1854524, a_q=-302842),
    16: dict(frac=8, one=256, inv_sqrt2=181, neg_b=453, a_q=-74),
}


def fixed_point_ibert_gelu(x_float, w):
    c = IBERT_CONSTS[w]
    frac, one, inv_sqrt2, neg_b, a_q = c["frac"], c["one"], c["inv_sqrt2"], c["neg_b"], c["a_q"]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    half, half2 = 1 << (frac - 1), 1 << frac
    xq = max(minv, min(maxv, round(x_float * (1 << frac))))

    u = max(minv, min(maxv, (xq * inv_sqrt2 + half) >> frac))
    mag = -u if u < 0 else u
    mag_clipped = min(mag, neg_b)
    term = mag_clipped - neg_b
    sq = (term * term + half) >> frac
    poly = ((a_q * sq + half) >> frac) + one
    l_term = 0 if xq == 0 else (-poly if xq < 0 else poly)
    one_plus_l = one + l_term
    y = max(minv, min(maxv, (xq * one_plus_l + half2) >> (frac + 1)))
    return y / (1 << frac)


def run_ibert_for_width(w, rng):
    max_abs_err, sum_abs_err, n, worst = 0.0, 0.0, 0, None
    for _ in range(50000):
        x = rng.uniform(-10, 10)
        got = fixed_point_ibert_gelu(x, w)
        exp = true_gelu(x)
        err = abs(got - exp)
        sum_abs_err += err
        n += 1
        if err > max_abs_err:
            max_abs_err, worst = err, (x, got, exp)
    print(f"ibert_gelu W={w}: n={n} mean_abs_err={sum_abs_err / n:.7f} max_abs_err={max_abs_err:.7f}")
    print("  worst case x =", round(worst[0], 4), " fixed-point:", round(worst[1], 5),
          " true GELU:", round(worst[2], 5))


def main():
    rng = random.Random(61)
    for w in (32, 16):
        run_for_width(w, rng)
    for w in (32, 16):
        run_ibert_for_width(w, rng)


if __name__ == "__main__":
    main()
