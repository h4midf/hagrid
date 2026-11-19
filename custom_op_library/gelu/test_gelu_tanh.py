"""Functional check of gelu_tanh.v against a Python model of the same fixed-point
arithmetic (bit-for-bit, including the 256-segment piecewise-linear table at each
width), as test_gelu.py does for gelu.v; the tables are computed here with the same
formula as rtl/gelu_tanh.v's (the tanh-form GELU at each segment's breakpoints,
independently at each width's scale). Each vector is evaluated by Yosys (eval) on
random and edge-case inputs at both characterized widths.

Also reports the fit's error against the true tanh-form GELU (the approximation
quality, as approximation_error.py does for gelu.v). Exit code 0 only if every vector
matches the model.
"""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "gelu_tanh.v")
N_SEG = 256
DOMAIN_HALF = 8

WIDTH_CFG = {
    32: dict(frac=20, shift=16),
    16: dict(frac=8, shift=4),
}


def true_gelu_tanh(x):
    return 0.5 * x * (1 + math.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * x ** 3)))


def build_table(scale):
    seg_width = (2 * DOMAIN_HALF) / N_SEG
    offs, slopes = [], []
    for k in range(N_SEG):
        x_lo = -DOMAIN_HALF + k * seg_width
        x_hi = x_lo + seg_width
        y_lo, y_hi = true_gelu_tanh(x_lo), true_gelu_tanh(x_hi)
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
    dom_q = DOMAIN_HALF * (1 << cfg["frac"])
    offs, slopes = TABLES[w]
    if x < -dom_q:
        return 0
    if x >= dom_q:
        return x
    s = x + dom_q
    idx = s >> cfg["shift"]
    intra = s & ((1 << cfg["shift"]) - 1)
    return offs[idx] + ((slopes[idx] * intra) >> cfg["shift"])


def vectors(w, n, rng):
    scale = 1 << WIDTH_CFG[w]["frac"]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    dom_q = DOMAIN_HALF * scale
    lo, hi = max(minv, -10 * scale), min(maxv, 10 * scale)
    v = [rng.randint(lo, hi) for _ in range(n)]
    v += [0, 1, -1, dom_q, -dom_q, dom_q - 1, -dom_q + 1, maxv, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} gelu_tanh",
             "hierarchy -top gelu_tanh", "proc -norom", "flatten", "select gelu_tanh"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def approximation_error(w, rng, n=50000):
    """(mean, max) |fit - true tanh GELU| over x uniform in [-10, 10]."""
    scale = 1 << WIDTH_CFG[w]["frac"]
    errs = []
    for _ in range(n):
        x = rng.uniform(-10, 10)
        got = ref(round(x * scale), w) / scale
        exp = true_gelu_tanh(x) if abs(x) < DOMAIN_HALF else (0.0 if x < 0 else x)
        errs.append(abs(got - exp))
    return sum(errs) / len(errs), max(errs)


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
        mean_err, max_err = approximation_error(w, rng)
        print(f"gelu_tanh W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}   "
              f"vs true tanh GELU: mean abs err {mean_err:.1e}, max abs err {max_err:.1e}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
