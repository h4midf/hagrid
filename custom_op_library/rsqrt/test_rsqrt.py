"""Functional check of the rsqrt circuits (../layer_norm/rtl/rsqrt_q12_20.v, rsqrt_q8_8.v:
seed table + one Newton iteration) against their Python model
(../layer_norm/test_layer_norm.py rsqrt_fixed), bit-exact by Yosys eval, on every bit
length of the input and random values. Also reports the relative error against the true
1/sqrt(x) over the inputs a layer norm's var + eps takes. Exit code 0 only if all pass."""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
LAYER_NORM = os.path.join(HERE, "..", "layer_norm")
sys.path.insert(0, LAYER_NORM)
from test_layer_norm import WIDTH_CFG, rsqrt_fixed  # noqa: E402

MODULES = {32: "rsqrt_q12_20", 16: "rsqrt_q8_8"}


def run(w, values):
    mod = MODULES[w]
    # proc -norom: keep the seed table as logic, which eval can evaluate
    lines = [f"read_verilog -sv {os.path.join(LAYER_NORM, 'rtl', mod + '.v')}", f"hierarchy -top {mod}",
             "proc -norom", "flatten"]
    lines += [f"eval -set val {w}'d{v} -show rsqrt_out" for v in values]
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    raw = []
    for text in re.findall(r"Eval result: \\rsqrt_out = ([^\n]+)\.", out):
        bits = re.match(r"(\d+)'([01]+)$", text)   # printed as bits or as a decimal
        raw.append(int(bits.group(2), 2) if bits else int(text))
    return [r - (1 << w) if r >= (1 << (w - 1)) else r for r in raw]


def inputs(w, rng):
    vals = []
    for bits in range(1, w):              # positive inputs only: every bit length, its ends, random
        lo, hi = 1 << (bits - 1), (1 << bits) - 1
        vals += [lo, hi] + [rng.randint(lo, hi) for _ in range(6)]
    return vals


def main():
    rng = random.Random(11)
    ok = True
    for w in (32, 16):
        frac = WIDTH_CFG[w]["frac"]
        vals = inputs(w, rng)
        got = run(w, vals)
        exp = [rsqrt_fixed(v, w) for v in vals]
        bad = [(v, g, e) for v, g, e in zip(vals, got, exp) if g != e]
        # relative error vs the true 1/sqrt over var + eps of a layer norm: 1e-5 .. 1e3
        worst = 0.0
        for v in vals:
            x = v / (1 << frac)
            if 1e-5 <= x <= 1e3:
                worst = max(worst, abs(rsqrt_fixed(v, w) / (1 << frac) * math.sqrt(x) - 1.0))
        passed = len(got) == len(vals) and not bad
        ok &= passed
        print(f"{MODULES[w]}: evaluated {len(got)}/{len(vals)}  mismatches {len(bad)}  "
              f"max relative error vs 1/sqrt(x) on [1e-5, 1e3]: {worst:.2%}  {'OK' if passed else 'FAIL'}")
        for v, g, e in bad[:3]:
            print("   val", v, "rtl", g, "ref", e)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
