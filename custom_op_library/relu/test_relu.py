"""Functional check of relu.v against max(x, 0). Yosys eval on random and edge-case
inputs, at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "relu.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref(x):
    return max(0, x)


def vectors(w, n, rng):
    v = [rng.randint(-(1 << (w - 1)), (1 << (w - 1)) - 1) for _ in range(n)]
    v += [0, 1, -1, (1 << (w - 1)) - 1, -(1 << (w - 1))]
    return v


def run(w, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} relu", "hierarchy -top relu", "proc"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(23)
    ok_all = True
    for w in (32, 16):
        vecs = vectors(w, 150, rng)
        got = run(w, vecs)
        exp = [ref(x) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"relu W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
