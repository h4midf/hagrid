"""Error of QUARK's appro_exp / appro_ln against true exp() / ln(), at both widths
(pure Python, no RTL; the same arithmetic as run_scripts/measure_accuracy/
quark_bert_large.py).
"""
import math
import random

WIDTH_CFG = {32: dict(frac=20, c0e=179625, c1e=699896, c2e=1046478,
                       c0l=-353290, c1l=2091663, c2l=-1730150),
             16: dict(frac=8, c0e=44, c1e=171, c2e=256,
                       c0l=-86, c1l=511, c2l=-422)}


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def fixed_point_exp(x_float, w):
    cfg = WIDTH_CFG[w]
    frac = cfg["frac"]
    scale = 1 << frac
    half = 1 << (frac - 1)
    x = round(x_float * scale)
    xshift = x + (x >> 1) - (x >> 4)
    floor_part = xshift >> frac
    low_bits = xshift & (scale - 1)
    qI = floor_part + (1 if low_bits != 0 else 0)
    qF = xshift - (qI << frac)
    qF2 = (qF * qF + half) >> frac
    c0qf2 = (cfg["c0e"] * qF2 + half) >> frac
    c1qf = (cfg["c1e"] * qF + half) >> frac
    xa = clampW(c0qf2 + c1qf + cfg["c2e"], w)
    shift_amt = -qI
    y = 0 if shift_amt >= w else (xa >> shift_amt)
    return y / scale


def fixed_point_ln(x_float, w):
    cfg = WIDTH_CFG[w]
    frac = cfg["frac"]
    scale = 1 << frac
    half = 1 << (frac - 1)
    x = round(x_float * scale)
    p = x.bit_length() - 1
    qM = p - frac
    frac_mask = scale - 1
    frac_bits = ((x >> (p - frac)) & frac_mask) if p >= frac else ((x << (frac - p)) & frac_mask)
    qN = scale + frac_bits
    qN2 = (qN * qN + half) >> frac
    c0qn2 = (cfg["c0l"] * qN2 + half) >> frac
    c1qn = (cfg["c1l"] * qN + half) >> frac
    poly = clampW(c0qn2 + c1qn + cfg["c2l"], w)
    log2x = clampW((qM << frac) + poly, w)
    y = log2x - (log2x >> 2) - (log2x >> 4)
    return y / scale


def run_for_width(w, rng):
    for name, f, true_f, domain in (
        ("appro_exp", fixed_point_exp, math.exp, lambda: rng.uniform(-15, 0)),
        ("appro_ln", fixed_point_ln, math.log, lambda: rng.uniform(0.01, 100)),
    ):
        max_err, sum_err, n, worst = 0.0, 0.0, 0, None
        for _ in range(20000):
            x = domain()
            got = f(x, w)
            true = true_f(x)
            err = abs(got - true)
            sum_err += err
            n += 1
            if err > max_err:
                max_err, worst = err, (x, got, true)
        print(f"{name} W={w}: n={n} mean_abs_err={sum_err / n:.6f} max_abs_err={max_err:.6f}")
        print("  worst case x =", round(worst[0], 4), " fixed-point:", round(worst[1], 5),
              " true:", round(worst[2], 5))


def main():
    rng = random.Random(139)
    for w in (32, 16):
        run_for_width(w, rng)


if __name__ == "__main__":
    main()
