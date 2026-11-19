"""Error of elu's fixed-point exp_pwl scheme against true ELU, at both widths
(W = 32 / Q12.20, W = 16 / Q8.8). Pure Python, no RTL (test_elu.py checks the RTL).
"""
import math
import random

TABLES = {
    32: dict(FRAC=20, LOG2E_Q=1512775, N_MAX=21,
             OFFSET=[1048576, 1004120, 961548, 920782, 881744, 844361, 808563, 774282,
                     741455, 710020, 679917, 651091, 623487, 597053, 571740, 547500],
             SLOPE=[-44456, -42572, -40766, -39038, -37383, -35798, -34281, -32827,
                    -31435, -30103, -28826, -27604, -26434, -25313, -24240, -23212]),
    16: dict(FRAC=8, LOG2E_Q=369, N_MAX=9,
             OFFSET=[256, 245, 235, 225, 215, 206, 197, 189, 181, 173, 166, 159, 152, 146, 140, 134],
             SLOPE=[-11, -10, -10, -10, -9, -9, -8, -8, -8, -7, -7, -7, -6, -6, -6, -6]),
}


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def exp_pwl(x, w):
    t = TABLES[w]
    frac = t["FRAC"]
    y_full = x * t["LOG2E_Q"]
    y = y_full >> frac
    neg_y = -y
    n_raw = neg_y >> frac
    r_frac = neg_y & ((1 << frac) - 1)
    seg = r_frac >> (frac - 4)
    intra = r_frac & ((1 << (frac - 4)) - 1)
    off, slope = t["OFFSET"][seg], t["SLOPE"][seg]
    g_full = off + (slope * intra) // (1 << (frac - 4))
    if n_raw >= t["N_MAX"]:
        return 0
    return g_full >> n_raw


def fixed_point_elu(x_float, w, alpha=1.0):
    frac = TABLES[w]["FRAC"]
    scale = 1 << frac
    xq = round(x_float * scale)
    alpha_q = round(alpha * scale)
    if xq > 0:
        return clamp(xq, w) / scale
    e = exp_pwl(xq, w)
    term = e - scale
    half = 1 << (frac - 1)
    p = term * alpha_q
    return clamp((p + half) >> frac, w) / scale


def true_elu(x, alpha=1.0):
    return x if x > 0 else alpha * (math.exp(x) - 1)


def run_for_width(w, rng):
    max_abs_err = 0.0
    sum_abs_err = 0.0
    n = 0
    worst = None
    for _ in range(20000):
        x = rng.uniform(-20, 20)
        got = fixed_point_elu(x, w)
        exp = true_elu(x)
        err = abs(got - exp)
        sum_abs_err += err
        n += 1
        if err > max_abs_err:
            max_abs_err = err
            worst = (x, got, exp)
    print(f"W={w}: n={n} mean_abs_err={sum_abs_err / n:.6f} max_abs_err={max_abs_err:.6f}")
    print("  worst case x =", worst[0], " fixed-point:", round(worst[1], 4), " true ELU:", round(worst[2], 4))

    max_abs_err_c = 0.0
    sum_abs_err_c = 0.0
    n2 = 0
    for _ in range(20000):
        x = rng.uniform(-4, 4)
        got = fixed_point_elu(x, w)
        exp = true_elu(x)
        err = abs(got - exp)
        sum_abs_err_c += err
        n2 += 1
        max_abs_err_c = max(max_abs_err_c, err)
    print(f"  clustered (x in [-4,4]): mean_abs_err={sum_abs_err_c / n2:.6f} max_abs_err={max_abs_err_c:.6f}")


def main():
    rng = random.Random(37)
    for w in (32, 16):
        run_for_width(w, rng)


if __name__ == "__main__":
    main()
