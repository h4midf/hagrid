"""Functional check of softmax_row8 against a Python model of the same fixed-point
arithmetic (bit for bit). Yosys eval on random and edge-case 8-bit score rows. Exit
code 0 only if every vector matches.
"""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "softmax_row8.v")

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


def ref_softmax_row8(s):
    max_val = max(s)
    d = [to_signed(si - max_val, 9) for si in s]
    x = [to_signed(di << 8, 24) for di in d]
    e = [exp_pwl16(xi) for xi in x]
    exp_sum = sum(e)  # always > 0, comfortably within width, no masking needed
    recip = to_signed(0 if exp_sum == 0 else (65536 // exp_sum), 16)
    p = [to_signed((ei * recip) >> 8, 16) for ei in e]
    return p


def vectors(n, rng):
    v = []
    for _ in range(n):
        v.append(tuple(rng.randint(0, 255) for _ in range(8)))
    # edge cases: all-tied, one dominant, monotonic ramp, all-zero, all-max
    v.append((100,) * 8)
    v.append((0, 255, 0, 0, 0, 0, 0, 0))
    v.append(tuple(range(0, 256, 32)))
    v.append((0,) * 8)
    v.append((255,) * 8)
    return v


def run(vecs):
    outs = [f"p{i}" for i in range(8)]
    lines = [f"read_verilog -sv {RTL}", "hierarchy -top softmax_row8", "proc -norom", "flatten", "select softmax_row8"]
    for s in vecs:
        sets = " ".join(f"-set s{i} 8'd{si}" for i, si in enumerate(s))
        shows = " ".join(f"-show {o}" for o in outs)
        lines.append(f"eval {sets} {shows}")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    cur = {}
    for name, bits in re.findall(r"Eval result: \\(p\d) = \d+'([01]+)", out):
        cur[name] = to_signed(int(bits, 2), len(bits))
        if len(cur) == 8:
            results.append(tuple(cur[f"p{i}"] for i in range(8)))
            cur = {}
    return results, out


def main():
    rng = random.Random(11)
    vecs = vectors(150, rng)
    got, raw_out = run(vecs)
    if len(got) != len(vecs):
        print(raw_out[-3000:])
    exp = [tuple(ref_softmax_row8(list(s))) for s in vecs]
    bad = [(s, g, e) for s, g, e in zip(vecs, got, exp) if g != e]
    status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
    print(f"softmax_row8  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
    for s, g, e in bad[:5]:
        print("   s =", s, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
