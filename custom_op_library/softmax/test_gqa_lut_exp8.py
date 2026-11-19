"""Functional check of gqa_lut_exp8.v against a Python model of the same fixed-point
arithmetic. Yosys eval on random and edge-case inputs <= 0, including values below
-8 (output 0), at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gqa_lut_exp8.v")

TABLES = {
    32: dict(frac=20, dom_lo=-8388608,
             bp=[-6160384, -3866624, -2686976, -1769472, -1048576, -589824, -262144],
             slope=[1128, 9839, 47971, 127934, 275262, 480592, 699413, 922184],
             icept=[0, 65536, 196608, 393216, 655360, 851968, 983040, 1048576]),
    16: dict(frac=8, dom_lo=-2048,
             bp=[-1504, -944, -656, -432, -256, -144, -64],
             slope=[0, 2, 12, 31, 67, 117, 171, 225],
             icept=[0, 16, 48, 96, 160, 208, 240, 256]),
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
    frac, dom_lo, bp, slope, icept = t["frac"], t["dom_lo"], t["bp"], t["slope"], t["icept"]
    if x < dom_lo:
        return 0
    half = 1 << (frac - 1)
    idx = 0
    while idx < len(bp) and x >= bp[idx]:
        idx += 1
    p = slope[idx] * x
    p_r = (p + half) >> frac
    return clampW(p_r + icept[idx], w)


def vectors(w, n, rng):
    t = TABLES[w]
    minv = -(1 << (w - 1))
    v = [rng.randint(minv, 0) for _ in range(n)]
    for b in t["bp"]:
        v += [b - 1, b, b + 1]
    v += [t["dom_lo"] - 1, t["dom_lo"], t["dom_lo"] + 1, 0, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} gqa_lut_exp8",
             "hierarchy -top gqa_lut_exp8", "proc -norom", "flatten", "select gqa_lut_exp8"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(167)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, 100, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"gqa_lut_exp8 W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
