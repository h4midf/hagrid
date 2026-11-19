"""Functional check of quark_softmax_row8.v against a Python model of the same
fixed-point arithmetic (QUARK's Eq.2 log-sum-exp form over the appro_exp and appro_ln
models). Yosys eval on random and edge-case 8-element rows, at both widths. Exit code
0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", f) for f in
       ("quark_appro_exp.v", "quark_appro_ln.v", "reduce_tree.v", "quark_softmax_row8.v")]
WIDTH_CFG = {32: dict(frac=20), 16: dict(frac=8)}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def appro_exp_ref(x, w, frac):
    half = 1 << (frac - 1)
    c0, c1, c2 = {32: (179625, 699896, 1046478), 16: (44, 171, 256)}[w]
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
    return 0 if shift_amt >= w else to_signed(xa >> shift_amt, w)


def appro_ln_ref(x, w, frac):
    half = 1 << (frac - 1)
    c0, c1, c2 = {32: (-353290, 2091663, -1730150), 16: (-86, 511, -422)}[w]
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


def ref_row(scores, w, frac):
    max_s = max(scores)
    ds = [to_signed(clampW(s - max_s, w), w) for s in scores]
    es = [appro_exp_ref(d, w, frac) for d in ds]
    s_sum = max(sum(es), 1)
    ln_s = appro_ln_ref(s_sum, w, frac)
    zs = []
    for d in ds:
        zf = d - ln_s
        z = clampW(zf, w)
        if z > 0:
            z = 0
        zs.append(z)
    return [appro_exp_ref(z, w, frac) for z in zs]


def pack_flat(vals, w, bits):
    v = 0
    for i, val in enumerate(vals):
        v |= (val & ((1 << bits) - 1)) << (i * bits)
    return v


def vectors(w, frac, n, rng):
    maxv = (1 << (w - 1)) - 1
    scale = 1 << frac
    v = [tuple(rng.randint(-10 * scale, maxv // 4) for _ in range(8)) for _ in range(n)]
    v.append((0,) * 8)
    v.append(tuple(range(8)))
    return v


def run(w, frac, rows):
    lines = [f"read_verilog -sv {' '.join(RTL)}", f"chparam -set W {w} -set FRAC {frac} quark_softmax_row8",
             "hierarchy -top quark_softmax_row8", "proc -norom", "flatten", "select quark_softmax_row8"]
    for row in rows:
        sets = " ".join(f"-set s{i} {w}'d{row[i] & ((1 << w) - 1)}" for i in range(8))
        shows = " ".join(f"-show p{i}" for i in range(8))
        lines.append(f"eval {sets} {shows}")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    cur = {}
    for name, raw in re.findall(r"Eval result: \\(p\d) = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        cur[name] = to_signed(int(m.group(1), 2), w) if m else int(raw)
        if len(cur) == 8:
            results.append([cur[f"p{i}"] for i in range(8)])
            cur = {}
    return results


def main():
    rng = random.Random(127)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        rows = vectors(w, frac, 40, rng)
        got = run(w, frac, rows)
        exp = [ref_row(row, w, frac) for row in rows]
        bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
        status = "OK" if len(got) == len(rows) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_softmax_row8 W={w:2d}  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  {status}")
        for r, g, e in bad[:3]:
            print("   scores =", r, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
