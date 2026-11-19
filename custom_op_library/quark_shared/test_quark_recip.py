"""Functional check of quark_recip.v against a Python model of the same fixed-point
arithmetic (seed table + 2 Newton iterations). Yosys eval on random and edge-case
positive inputs, at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "quark_recip.v")
IDX_BITS = 5
WIDTH_CFG = {32: dict(frac=20), 16: dict(frac=8)}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def build_seed_table(scale):
    return [round((1.0 / (1.0 + idx / (1 << IDX_BITS))) * scale) for idx in range(1 << IDX_BITS)]


SEEDS = {w: build_seed_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def ref(x, w):
    frac = WIDTH_CFG[w]["frac"]
    half = 1 << (frac - 1)
    two = 2 << frac
    p = x.bit_length() - 1
    e = p - frac
    idx = (x >> (p - IDX_BITS)) & ((1 << IDX_BITS) - 1) if p >= IDX_BITS else (x << (IDX_BITS - p)) & ((1 << IDX_BITS) - 1)
    seed = SEEDS[w][idx]
    y0 = (seed >> e) if e >= 0 else (seed << (-e))

    def newton(y):
        xy = (x * y + half) >> frac
        t = clampW(two - xy, w)
        y_new = (y * t + half) >> frac
        return clampW(y_new, w)

    y1 = newton(y0)
    y2 = newton(y1)
    return y2


def vectors(w, frac, n, rng):
    # x must stay in the signed positive range, and above the floor where
    # y0 = seed << (-e) overflows W bits (RTL header); min_x is clear of it.
    maxv = (1 << (w - 1)) - 1
    scale = 1 << frac
    min_x = max(1, scale // 8)  # comfortable margin above the overflow boundary (scale//100 was too tight for W=16)
    v = [rng.randint(min_x, maxv) for _ in range(n)]
    v += [min_x, scale, scale // 2, min(200 * scale, maxv), maxv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} quark_recip",
             "hierarchy -top quark_recip", "proc -norom", "flatten", "select quark_recip"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(109)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 150, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_recip W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
