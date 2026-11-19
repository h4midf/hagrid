"""Functional check of quark_appro_exp.v against a Python model of the same fixed-point
arithmetic (QUARK's Eq.3-5). Yosys eval on random and edge-case non-positive inputs,
at both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "quark_appro_exp.v")
CONSTS = {
    32: dict(frac=20, c0=179625, c1=699896, c2=1046478),
    16: dict(frac=8, c0=44, c1=171, c2=256),
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
    c = CONSTS[w]
    frac, c0, c1, c2 = c["frac"], c["c0"], c["c1"], c["c2"]
    half = 1 << (frac - 1)
    xshift = x + (x >> 1) - (x >> 4)
    floor_part = xshift >> frac
    low_bits = xshift & ((1 << frac) - 1)
    qI = floor_part + (1 if low_bits != 0 else 0)
    qF = xshift - (qI << frac)
    qF2 = (qF * qF + half) >> frac
    c0qf2 = (c0 * qF2 + half) >> frac
    c1qf = (c1 * qF + half) >> frac
    xa = clampW(c0qf2 + c1qf + c2, w)
    shift_amt = -qI
    return 0 if shift_amt >= w else (xa >> shift_amt)


def vectors(w, frac, n, rng):
    # MINV excluded: xshift = x+x/2-x/16 overflows W bits there (RTL header)
    scale = 1 << frac
    v = [rng.randint(-15 * scale, 0) for _ in range(n)]
    v += [0, -1, -scale, -2 * scale, -10 * scale]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} quark_appro_exp",
             "hierarchy -top quark_appro_exp", "proc -norom", "flatten", "select quark_appro_exp"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(103)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 150, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_appro_exp W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
