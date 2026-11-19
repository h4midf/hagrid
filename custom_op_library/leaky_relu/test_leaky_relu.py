"""Functional check of leaky_relu.v against a Python model, at three alpha values and
both widths (Q12.20, Q8.8). Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "leaky_relu.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def ref(x, alpha, frac, w):
    if x >= 0:
        return x
    half = 1 << (frac - 1)
    return clamp((x * alpha + half) >> frac, w)


def vectors(w, n, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    v = [rng.randint(minv, maxv) for _ in range(n)]
    v += [0, 1, -1, maxv, minv]
    return v


def run(w, frac, alpha, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} -set ALPHA {alpha & ((1<<w)-1)} "
              "leaky_relu", "hierarchy -top leaky_relu", "proc"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(29)
    ok_all = True
    # per width: 0.01, 0.125 and 1.0 (identity)
    cases = {
        32: (("0.01", 10486), ("0.125", 1 << 17), ("1.0", 1 << 20)),
        16: (("0.01", 3), ("0.125", 1 << 5), ("1.0", 1 << 8)),
    }
    for w, frac in ((32, 20), (16, 8)):
        for alpha_name, alpha in cases[w]:
            vecs = vectors(w, 150, rng)
            got = run(w, frac, alpha, vecs)
            exp = [ref(x, alpha, frac, w) for x in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"leaky_relu W={w:2d} alpha={alpha_name:6s}  evaluated {len(got)}/{len(vecs)}  "
                  f"mismatches {len(bad)}  {status}")
            for v, g, e in bad[:5]:
                print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
