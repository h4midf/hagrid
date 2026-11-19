"""Functional check of wang_tcad_gelu24.v (24-entry nonuniform-LUT GELU, breakpoints
calibrated on bert-large) against a Python model of the same fixed-point arithmetic, not
true GELU. Yosys eval on random and edge-case inputs (around every bin boundary and the
+-3 domain edges), at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "wang_tcad_gelu24.v")

TABLES = {
    32: dict(frac=20, dom=3145728,
             bp=[-2916352, -2654208, -2424832, -2195456, -1933312, -1703936, -1474560, -1212416, -950272, -688128,
                 -393216, -131072, 131072, 425984, 688128, 950272, 1245184, 1507328, 1769472, 2064384, 2326528,
                 2588672, 2883584],
             out=[0, 0, -32768, -32768, -65536, -65536, -98304, -131072, -163840, -163840, -163840, -98304, 0,
                  163840, 393216, 655360, 950272, 1245184, 1540096, 1867776, 2162688, 2424832, 2719744, 3014656]),
    16: dict(frac=8, dom=768,
             bp=[-712, -648, -592, -536, -472, -416, -360, -296, -232, -168, -96, -32, 32, 104, 168, 232, 304, 368,
                 432, 504, 568, 632, 704],
             out=[0, 0, -8, -8, -16, -16, -24, -32, -40, -40, -40, -24, 0, 40, 96, 160, 232, 304, 376, 456, 528, 592,
                  664, 736]),
}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref(x, w):
    t = TABLES[w]
    dom = t["dom"]
    if x < -dom:
        return 0
    if x > dom:
        return x
    bp = t["bp"]
    idx = 0
    while idx < len(bp) and x >= bp[idx]:
        idx += 1
    return t["out"][idx]


def vectors(w, n, rng):
    t = TABLES[w]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    v = [rng.randint(minv, maxv) for _ in range(n)]
    for b in t["bp"]:
        v += [b - 1, b, b + 1]
    v += [-t["dom"] - 1, -t["dom"], -t["dom"] + 1, t["dom"] - 1, t["dom"], t["dom"] + 1, maxv, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} wang_tcad_gelu24",
             "hierarchy -top wang_tcad_gelu24", "proc -norom", "flatten", "select wang_tcad_gelu24"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(157)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, 100, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"wang_tcad_gelu24 W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
