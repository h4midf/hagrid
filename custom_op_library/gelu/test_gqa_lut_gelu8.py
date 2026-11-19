"""Functional check of gqa_lut_gelu8.v (8-entry GA-searched piecewise-linear GELU)
against a Python model of the same fixed-point arithmetic, not true GELU. Yosys eval on
random and edge-case inputs, including values far outside [-4, 4] (extrapolated, not
clamped), at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gqa_lut_gelu8.v")

TABLES = {
    32: dict(frac=20,
             bp=[-2883584, -2293760, -1146880, -786432, -262144, 491520, 2359296],
             slope=[-5807, -42184, -117730, -65924, 143396, 610830, 1144232, 1062282],
             icept=[-32768, -131072, -294912, -229376, -98304, 0, -262144, -65536]),
    16: dict(frac=8,
             bp=[-704, -560, -280, -192, -64, 120, 576],
             slope=[-1, -10, -29, -16, 35, 149, 279, 259],
             icept=[-8, -32, -72, -56, -24, 0, -64, -16]),
}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def ref(x, w):
    t = TABLES[w]
    frac, bp, slope, icept = t["frac"], t["bp"], t["slope"], t["icept"]
    half = 1 << (frac - 1)
    idx = 0
    while idx < len(bp) and x >= bp[idx]:
        idx += 1
    p = slope[idx] * x
    p_r = (p + half) >> frac
    return clampW(p_r + icept[idx], w)


def vectors(w, n, rng):
    t = TABLES[w]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    v = [rng.randint(minv, maxv) for _ in range(n)]
    for b in t["bp"]:
        v += [b - 1, b, b + 1]
    v += [maxv, minv, 0]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} gqa_lut_gelu8",
             "hierarchy -top gqa_lut_gelu8", "proc -norom", "flatten", "select gqa_lut_gelu8"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(163)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, 100, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"gqa_lut_gelu8 W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
