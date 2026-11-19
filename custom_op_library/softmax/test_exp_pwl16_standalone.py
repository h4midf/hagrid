"""Functional check of exp_pwl16_standalone.v against test_softmax.py's exp_pwl16
model. Yosys eval on random and edge-case x values. Exit code 0 only if every vector
matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "exp_pwl16_standalone.v")

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


def run(xs):
    lines = [f"read_verilog -sv {RTL}", "hierarchy -top exp_pwl16_standalone", "proc -norom", "flatten",
             "select exp_pwl16_standalone"]
    for x in xs:
        lines.append(f"eval -set x 24'd{x & 0xFFFFFF} -show exp_out")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\exp_out = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        results.append(to_signed(int(m.group(2), 2), int(m.group(1))) if m else to_signed(int(raw), 16))
    return results


def main():
    rng = random.Random(23)
    xs = [-rng.randint(0, 4000) for _ in range(60)] + [0, -1, -256, -2304, -2305, -65536]
    got = run(xs)
    exp = [exp_pwl16(x) for x in xs]
    bad = [(x, g, e) for x, g, e in zip(xs, got, exp) if g != e]
    status = "OK" if len(got) == len(xs) and not bad else "FAIL"
    print(f"exp_pwl16_standalone  evaluated {len(got)}/{len(xs)}  mismatches {len(bad)}  {status}")
    for x, g, e in bad[:5]:
        print("   x =", x, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
