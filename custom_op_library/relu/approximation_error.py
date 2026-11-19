"""Error of Shifted-ReLU (relu(x)+relu(x-1)+relu(x-2)+relu(x-3)) against the true
exp() it replaces in HAMMER's softmax, at both widths (pure Python, no RTL).
"""
import math
import random


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def fixed_point_shifted_relu(x_float, w, frac):
    scale = 1 << frac
    xq = round(x_float * scale)
    one = scale
    s = max(0, xq) + max(0, xq - one) + max(0, xq - 2 * one) + max(0, xq - 3 * one)
    return clamp(s, w) / scale


def true_exp(x):
    return math.exp(x)


def run_for_width(w, frac, rng):
    max_abs_err, sum_abs_err, n, worst = 0.0, 0.0, 0, None
    for _ in range(20000):
        x = rng.uniform(-4, 4)
        got = fixed_point_shifted_relu(x, w, frac)
        exp = true_exp(x)
        err = abs(got - exp)
        sum_abs_err += err
        n += 1
        if err > max_abs_err:
            max_abs_err, worst = err, (x, got, exp)
    print(f"W={w}: n={n} mean_abs_err={sum_abs_err / n:.6f} max_abs_err={max_abs_err:.6f}")
    print("  worst case x =", round(worst[0], 4), " shifted-relu:", round(worst[1], 4),
          " true exp:", round(worst[2], 4))


def main():
    rng = random.Random(41)
    for w, frac in ((32, 20), (16, 8)):
        run_for_width(w, frac, rng)


if __name__ == "__main__":
    main()
