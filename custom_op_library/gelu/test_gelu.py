"""Functional check of gelu.v against a Python model of the same fixed-point arithmetic,
bit for bit, including the 256-segment table at each width (not against true GELU:
approximation_error.py). The OFFSET / SLOPE tables are computed here with rtl/gelu.v's
formula (GELU at each segment's breakpoints, at each width's scale).

Each vector is evaluated by Yosys (eval) on random and edge-case inputs,
at both characterized widths. Exit code 0 only if every vector matches.
"""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gelu.v")
N_SEG = 256
DOMAIN_HALF = 8

WIDTH_CFG = {
    32: dict(frac=20, shift=16),
    16: dict(frac=8, shift=4),
}


def true_gelu(x):
    return 0.5 * x * (1 + math.erf(x / math.sqrt(2)))


def build_table(scale):
    seg_width = (2 * DOMAIN_HALF) / N_SEG
    offs, slopes = [], []
    for k in range(N_SEG):
        x_lo = -DOMAIN_HALF + k * seg_width
        x_hi = x_lo + seg_width
        y_lo, y_hi = true_gelu(x_lo), true_gelu(x_hi)
        offs.append(round(y_lo * scale))
        slopes.append(round((y_hi - y_lo) * scale))
    return offs, slopes


TABLES = {w: build_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref(x, w):
    cfg = WIDTH_CFG[w]
    scale = 1 << cfg["frac"]
    dom_q = DOMAIN_HALF * scale
    offs, slopes = TABLES[w]
    if x < -dom_q:
        return 0
    if x >= dom_q:
        return x
    s = x + dom_q
    idx = s >> cfg["shift"]
    intra = s & ((1 << cfg["shift"]) - 1)
    off, slope = offs[idx], slopes[idx]
    return off + ((slope * intra) >> cfg["shift"])


def vectors(w, n, rng):
    scale = 1 << WIDTH_CFG[w]["frac"]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    dom_q = DOMAIN_HALF * scale
    lo, hi = max(minv, -10 * scale), min(maxv, 10 * scale)
    v = [rng.randint(lo, hi) for _ in range(n)]
    v += [0, 1, -1, dom_q, -dom_q, dom_q - 1, -dom_q + 1, maxv, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} gelu", "hierarchy -top gelu",
              "proc -norom", "flatten", "select gelu"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(59)
    ok_all = True
    for w, cfg in WIDTH_CFG.items():
        vecs = vectors(w, 150, rng)
        got = run(w, cfg["frac"], vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"gelu W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
