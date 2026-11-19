"""Functional check of gqa_lut_softmax_row8.v against a Python model of the same
fixed-point arithmetic. Yosys eval on random and edge-case score rows. Exit code 0
only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gqa_lut_softmax_row8.v")
EXP_RTL = os.path.join(HERE, "rtl", "gqa_lut_exp8.v")
BASE_RTL = os.path.join(HERE, "rtl", "softmax_row8.v")  # for max_reduce8

# gqa_lut_exp8.v's Q8.8 table
THRESH = [-1504, -944, -656, -432, -256, -144, -64]
SLOPE = [0, 2, 12, 31, 67, 117, 171, 225]
ICEPT = [0, 16, 48, 96, 160, 208, 240, 256]
DOM_LO = -2048
FRAC = 8
HALF = 1 << (FRAC - 1)
MAXV = 32767
MINV = -32768


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def gqa_lut_exp8_q8_8(x):
    idx = 7
    for i, t in enumerate(THRESH):
        if x < t:
            idx = i
            break
    slope, icept = SLOPE[idx], ICEPT[idx]
    p_full = slope * x
    p_r = to_signed(p_full + HALF, 32) >> FRAC  # arithmetic shift, matches >>> on a signed value
    y_full = p_r + icept
    y_lut = MAXV if y_full > MAXV else (MINV if y_full < MINV else y_full)
    return 0 if x < DOM_LO else y_lut


def ref_row(scores):
    max_val = max(scores)
    ds = [to_signed(si - max_val, 9) for si in scores]
    xs_wide = [to_signed(d << 8, 24) for d in ds]
    xs = [max(xw, -32768) for xw in xs_wide]  # same saturate-to-int16-min the RTL does before the LUT call
    es = [gqa_lut_exp8_q8_8(x) for x in xs]
    exp_sum = sum(es)
    recip = to_signed(0 if exp_sum == 0 else (65536 // exp_sum), 16)
    return [to_signed((ei * recip) >> 8, 16) for ei in es]


def vectors(n, rng):
    rows = [[rng.randint(0, 255) for _ in range(8)] for _ in range(n)]
    rows += [[0] * 8, [255] * 8, [255, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7], [255, 254, 1, 0, 0, 0, 0, 0]]
    return rows


def run(rows):
    lines = [f"read_verilog -sv {BASE_RTL}", f"read_verilog -sv {EXP_RTL}", f"read_verilog -sv {RTL}",
             "hierarchy -top gqa_lut_softmax_row8", "proc -norom", "flatten", "select gqa_lut_softmax_row8"]
    for row in rows:
        sets = " ".join(f"-set s{i} 8'd{row[i]}" for i in range(8))
        shows = " ".join(f"-show p{i}" for i in range(8))
        lines.append(f"eval {sets} {shows}")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    cur = {}
    for name, raw in re.findall(r"Eval result: \\(p\d) = ([^\n]+)\.", out):
        m = re.match(r"16'([01]+)$", raw)
        cur[name] = to_signed(int(m.group(1), 2), 16) if m else int(raw)
        if len(cur) == 8:
            results.append([cur[f"p{i}"] for i in range(8)])
            cur = {}
    return results


def main():
    rng = random.Random(97)
    rows = vectors(60, rng)
    got = run(rows)
    exp = [ref_row(row) for row in rows]
    bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
    status = "OK" if len(got) == len(rows) and not bad else "FAIL"
    print(f"gqa_lut_softmax_row8  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  {status}")
    for r, g, e in bad[:5]:
        print("   scores =", r, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
