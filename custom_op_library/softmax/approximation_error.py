"""Error of softmax_row8's fixed-point scheme (16-segment PWL exp) against true
softmax (pure Python, no RTL; the same arithmetic as test_softmax.py).
"""
import math
import random

LOG2E_Q8_8 = 369
OFFSET = [256, 245, 235, 225, 215, 206, 197, 189, 181, 173, 166, 159, 152, 146, 140, 134]
SLOPE = [-11, -10, -10, -10, -9, -9, -8, -8, -8, -7, -7, -7, -6, -6, -6, -6]
N_MAX = 9


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def exp_pwl16(x):
    y_full = x * LOG2E_Q8_8
    y = to_signed(y_full >> 8, 32)
    neg_y = -y
    n_raw = neg_y >> 8
    r_frac = neg_y & 0xFF
    seg = (r_frac >> 4) & 0xF
    intra = r_frac & 0xF
    off = OFFSET[seg]
    slope = SLOPE[seg]
    g_full = to_signed((off << 4) + slope * intra, 16)
    g_r = to_signed(g_full >> 4, 12)
    if n_raw >= N_MAX:
        return 0
    return g_r >> n_raw


def fixed_point_softmax(s):
    max_val = max(s)
    d = [to_signed(si - max_val, 9) for si in s]
    x = [to_signed(di << 8, 24) for di in d]
    e = [exp_pwl16(xi) for xi in x]
    exp_sum = sum(e)
    recip = to_signed(0 if exp_sum == 0 else (65536 // exp_sum), 16)
    p = [to_signed((ei * recip) >> 8, 16) for ei in e]
    return [pi / 256.0 for pi in p]


def true_softmax(s):
    m = max(s)
    e = [math.exp(si - m) for si in s]
    z = sum(e)
    return [ei / z for ei in e]


def main():
    rng = random.Random(3)
    max_abs_err = 0.0
    sum_abs_err = 0.0
    n = 0
    worst = None
    for _ in range(5000):
        s = [rng.randint(0, 255) for _ in range(8)]
        got = fixed_point_softmax(s)
        exp = true_softmax(s)
        for g, e in zip(got, exp):
            err = abs(g - e)
            sum_abs_err += err
            n += 1
            if err > max_abs_err:
                max_abs_err = err
                worst = (s, got, exp)

    print(f"n={n} mean_abs_err={sum_abs_err / n:.5f} max_abs_err={max_abs_err:.5f}")
    print("worst case s =", worst[0])
    print("  fixed-point:", [round(v, 4) for v in worst[1]])
    print("  true softmax:", [round(v, 4) for v in worst[2]])

    # scores clustered near the max; uniform 0..255 is close to the worst case
    max_abs_err_clustered = 0.0
    sum_abs_err_clustered = 0.0
    n2 = 0
    for _ in range(5000):
        base = rng.randint(0, 255)
        s = [max(0, min(255, base + rng.randint(-4, 4))) for _ in range(8)]
        got = fixed_point_softmax(s)
        exp = true_softmax(s)
        for g, e in zip(got, exp):
            err = abs(g - e)
            sum_abs_err_clustered += err
            n2 += 1
            max_abs_err_clustered = max(max_abs_err_clustered, err)
    print(f"\nclustered (scores within +-4 of each other): "
          f"mean_abs_err={sum_abs_err_clustered / n2:.5f} max_abs_err={max_abs_err_clustered:.5f}")


if __name__ == "__main__":
    main()
