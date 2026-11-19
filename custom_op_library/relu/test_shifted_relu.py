"""Functional check of shifted_relu.v against a Python model. Yosys eval on random
inputs and inputs around 0, ONE, 2*ONE and 3*ONE, at both widths. Exit code 0 only
if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "shifted_relu.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def ref(x, frac, w):
    one = 1 << frac
    s = max(0, x) + max(0, x - one) + max(0, x - 2 * one) + max(0, x - 3 * one)
    return clamp(s, w)


def vectors(w, frac, n, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    one = 1 << frac
    v = [rng.randint(minv, maxv) for _ in range(n)]
    for k in (0, one, 2 * one, 3 * one):
        v += [k - 1, k, k + 1]
    v += [0, 1, -1, maxv, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} shifted_relu",
             "hierarchy -top shifted_relu", "proc"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(31)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 150, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, frac, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"shifted_relu W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
