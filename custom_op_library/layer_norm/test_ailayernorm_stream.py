"""Functional check of rtl/ailayernorm_stream.v against a Python model of the same
fixed-point arithmetic (bit for bit): SOLE's per-lane PTF quantize and DynamicCompress,
the row sums divided by ROW through the reciprocal, per-element gamma / beta.

For every characterized width (Q12.20, Q8.8), lane count N (4, 8) and row length
(64, 768 and 1024), yosys simulates rtl/ailayernorm_stream_tb.v on random rows (one with
outliers, random alpha in [-6, 6]) and edge rows (all zero, all tied, a ramp with
mixed alpha signs). Exit code 0 only if every element of every row matches.

Usage (from the repo root):
  python3 custom_op_library/layer_norm/test_ailayernorm_stream.py
"""
import os
import random
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
FILES = [os.path.join(RTL, f) for f in ("reduce_tree.v", "rsqrt_q12_20.v", "rsqrt_q8_8.v", "row_div.v",
                                        "ailn_stream_lane.v", "ailn_var.v", "ailayernorm_stream.v",
                                        "ailayernorm_stream_tb.v")]
sys.path.insert(0, HERE)
from test_layer_norm import WIDTH_CFG, clampW, rsqrt_fixed, to_signed  # noqa: E402
from test_layer_norm_stream import LANES, ROWS, outputs, write_hex  # noqa: E402

RS = 40
EPS = 1


def rshift_round(v, amt):
    """(v + 2^(amt-1)) >> amt, as the RTL's half_dyn + >>> (amt > 0)."""
    half = 1 << (amt - 1)
    return (v + half) >> amt


def lane_fixed(x, alpha, w, frac):
    """One lane's PTF int8 quantize (x_q), x_q rescaled back to Q.FRAC, and the
    DynamicCompress square rescaled by 2*alpha."""
    shift_amt = alpha + frac
    if shift_amt >= 0:
        q_rshift = x if shift_amt == 0 else rshift_round(x, shift_amt)
    else:
        q_rshift = x << (-shift_amt)
    x_q = clampW(max(-128, min(127, q_rshift)), 8)

    if shift_amt >= 0:
        x_shifted = x_q << shift_amt
    else:
        x_shifted = rshift_round(x_q, -shift_amt) if (-shift_amt) > 0 else x_q
    x_shifted = clampW(x_shifted, w)

    mag = -x_q if x_q < 0 else x_q
    high = mag >= 64
    xc_raw = (mag >> 4) if high else (mag >> 2)
    xc = min(xc_raw, 15)
    xc2_raw = xc * xc
    xc2 = (xc2_raw << 4) if high else xc2_raw

    two_alpha = alpha * 2
    if two_alpha >= 0:
        xc2_shifted = xc2 << two_alpha
    else:
        xc2_shifted = rshift_round(xc2, -two_alpha) if (-two_alpha) > 0 else xc2
    xc2_shifted = clampW(xc2_shifted, w)

    return x_shifted, xc2_shifted


def square_shifted(x, alpha, frac):
    """ailn_stream_lane.v's decompressed square: lane_fixed's x_q and
    DynamicCompress, rescaled by 2*alpha + FRAC (Q.FRAC units), 2W bits, not clamped."""
    shift_amt = alpha + frac
    if shift_amt > 0:
        q = (x + (1 << (shift_amt - 1))) >> shift_amt
    elif shift_amt == 0:
        q = x
    else:
        q = x << (-shift_amt)
    x_q = max(-128, min(127, q))
    mag = -x_q if x_q < 0 else x_q
    high = mag >= 64
    xc = min((mag >> 4) if high else (mag >> 2), 15)
    xc2 = (xc * xc) << 4 if high else xc * xc
    sh = 2 * alpha + frac
    if sh >= 0:
        return xc2 << sh
    return (xc2 + (1 << (-sh - 1))) >> (-sh)


def ref(x, alpha, gamma, beta, eps, w):
    """ailayernorm_stream.v's arithmetic over one whole row."""
    frac = WIDTH_CFG[w]["frac"]
    half = 1 << (frac - 1)
    recip = ((1 << RS) + len(x) // 2) // len(x)
    xs = [lane_fixed(v, a, w, frac)[0] for v, a in zip(x, alpha)]
    xc2s = [square_shifted(v, a, frac) for v, a in zip(x, alpha)]
    ex = to_signed(((sum(xs) * recip) >> RS) & ((1 << w) - 1), w)
    ex2 = ((sum(xc2s) * recip) >> RS) << 4              # wide
    ex_sq = (ex * ex + half) >> frac                    # 2W bits
    var = max(clampW(ex2 - ex_sq, w), 1)
    var_eps = max(clampW(var + eps, w), 1)
    inv_std = rsqrt_fixed(var_eps, w)
    out = []
    for v, g, b in zip(xs, gamma, beta):
        diff = clampW(v - ex, w)
        norm = clampW((diff * inv_std + half) >> frac, w)
        scaled = clampW((norm * g + half) >> frac, w)
        out.append(clampW(scaled + b, w))
    return out


def simulate(x, alpha, gamma, beta, n, w, row):
    frac = WIDTH_CFG[w]["frac"]
    with tempfile.TemporaryDirectory() as tmp:
        write_hex(os.path.join(tmp, "x.hex"), x, n, w)
        write_hex(os.path.join(tmp, "alpha.hex"), alpha, n, 8)
        write_hex(os.path.join(tmp, "gamma.hex"), gamma, n, w)
        write_hex(os.path.join(tmp, "beta.hex"), beta, n, w)
        vcd_path = os.path.join(tmp, "out.vcd")
        cycles = 2 * (row // n) + 6
        script = (f"read_verilog -sv {' '.join(FILES)}; "
                  f"chparam -set N {n} -set W {w} -set FRAC {frac} -set ROW {row} -set EPS {EPS} "
                  f"ailayernorm_stream_tb; hierarchy -top ailayernorm_stream_tb; proc; flatten; opt_clean; "
                  f"sim -clock clk -zinit -n {cycles} -vcd {vcd_path}")
        subprocess.run(["yosys", "-q", "-p", script], check=True, cwd=tmp, stderr=subprocess.DEVNULL)
        with open(vcd_path) as f:
            return outputs(f.read(), n, w, "step")


def rows_for(row, w, rng):
    scale = 1 << WIDTH_CFG[w]["frac"]
    big = (1 << (w - 1)) - 1
    gamma = [rng.randint(scale // 4, 2 * scale) for _ in range(row)]
    beta = [rng.randint(-scale, scale) for _ in range(row)]
    alpha = [rng.randint(-6, 6) for _ in range(row)]
    rand = [rng.randint(-10 * scale, 10 * scale) for _ in range(row)]
    outl = list(rand)
    for k in rng.sample(range(row), 4):
        outl[k] = rng.choice((-1, 1)) * rng.randint(big // 4, big)
    mixed = [(2, -1, 0, 3)[k % 4] for k in range(row)]
    return [
        ("random", rand, alpha, gamma, beta),
        ("outliers", outl, alpha, gamma, beta),
        ("zero", [0] * row, [0] * row, [scale] * row, [0] * row),
        ("tied", [scale] * row, [0] * row, [scale] * row, [0] * row),
        ("ramp", list(range(1, row + 1)), mixed, [scale] * row, [0] * row),
    ]


def main():
    rng = random.Random(0)
    bad, total = [], 0
    for w in (32, 16):
        for n in LANES:
            for row in ROWS:
                for name, x, alpha, gamma, beta in rows_for(row, w, rng):
                    total += 1
                    got = simulate(x, alpha, gamma, beta, n, w, row)
                    exp = ref(x, alpha, gamma, beta, EPS, w)
                    ok = got == exp
                    print(f"  W={w} N={n} ROW={row} {name}: {'ok' if ok else 'MISMATCH'}", flush=True)
                    if not ok:
                        diff = [k for k in range(min(len(got), len(exp))) if got[k] != exp[k]]
                        bad.append((w, n, row, name, len(got), len(exp), diff[:5]))
    print(f"ailayernorm_stream: {total - len(bad)}/{total} rows match")
    for b in bad:
        print(f"  {b}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
