"""Functional check of ibert_gelu.v (I-BERT's Eq. 4-9 quadratic GELU) against a Python
model of the same fixed-point arithmetic, not true GELU. Includes x = 0, which the RTL
special-cases. Yosys eval on random and edge-case inputs, at both widths. Exit code 0
only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "ibert_gelu.v")

CONSTS = {
    32: dict(frac=20, one=1048576, inv_sqrt2=741455, neg_b=1854524, a_q=-302842),
    16: dict(frac=8, one=256, inv_sqrt2=181, neg_b=453, a_q=-74),
}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def ref(x, w):
    c = CONSTS[w]
    frac, one, inv_sqrt2, neg_b, a_q = c["frac"], c["one"], c["inv_sqrt2"], c["neg_b"], c["a_q"]
    half, half2 = 1 << (frac - 1), 1 << frac

    u = clamp((x * inv_sqrt2 + half) >> frac, w)
    mag = -u if u < 0 else u
    mag_clipped = min(mag, neg_b)
    term = mag_clipped - neg_b
    sq = (term * term + half) >> frac
    poly = ((a_q * sq + half) >> frac) + one
    l_term = 0 if x == 0 else (-poly if x < 0 else poly)
    one_plus_l = one + l_term
    y = clamp((x * one_plus_l + half2) >> (frac + 1), w)
    return y


def vectors(w, n, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    v = [rng.randint(minv, maxv) for _ in range(n)]
    v += [0, 1, -1, maxv, minv]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} ibert_gelu",
             "hierarchy -top ibert_gelu", "proc"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(53)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, 150, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"ibert_gelu W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
