"""Functional check of quark_appro_ln.v against a Python model of the same fixed-point
arithmetic (QUARK's Eq.6-9). Yosys eval on random and edge-case positive inputs, at
both widths. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "quark_appro_ln.v")
CONSTS = {
    32: dict(frac=20, c0=-353290, c1=2091663, c2=-1730150),
    16: dict(frac=8, c0=-86, c1=511, c2=-422),
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
    p = x.bit_length() - 1
    qM = p - frac
    frac_mask = (1 << frac) - 1
    frac_bits = ((x >> (p - frac)) & frac_mask) if p >= frac else ((x << (frac - p)) & frac_mask)
    qN = (1 << frac) + frac_bits
    qN2 = (qN * qN + half) >> frac
    c0qn2 = (c0 * qN2 + half) >> frac
    c1qn = (c1 * qN + half) >> frac
    poly = clampW(c0qn2 + c1qn + c2, w)
    log2x = clampW((qM << frac) + poly, w)
    return to_signed(log2x - (log2x >> 2) - (log2x >> 4), w)


def vectors(w, frac, n, rng):
    scale = 1 << frac
    v = [rng.randint(1, 100 * scale) for _ in range(n)]
    v += [1, 2, scale, scale // 2, 100 * scale, (1 << (w - 1)) - 1]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {w} -set FRAC {frac} quark_appro_ln",
             "hierarchy -top quark_appro_ln", "proc -norom", "flatten", "select quark_appro_ln"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(107)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 150, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_appro_ln W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
