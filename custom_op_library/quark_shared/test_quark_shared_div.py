"""Functional check of quark_shared_div.v against a Python model of the same
fixed-point arithmetic (QUARK Eq.18 over the appro_ln, appro_exp and recip models).
Yosys eval on random and edge-case positive (a, b) pairs, at both widths. Exit code 0
only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", f) for f in
       ("quark_appro_exp.v", "quark_appro_ln.v", "quark_recip.v", "quark_shared_div.v")]
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


def appro_exp_ref(x, w, frac):
    half = 1 << (frac - 1)
    c = {32: (179625, 699896, 1046478), 16: (44, 171, 256)}[w]
    c0, c1, c2 = c
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
    c = {32: (-353290, 2091663, -1730150), 16: (-86, 511, -422)}[w]
    c0, c1, c2 = c
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


def build_seed_table(scale):
    return [round((1.0 / (1.0 + idx / (1 << IDX_BITS))) * scale) for idx in range(1 << IDX_BITS)]


SEEDS = {w: build_seed_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def recip_ref(x, w, frac):
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
        return clampW((y * t + half) >> frac, w)

    return newton(newton(y0))


def ref(a, b, w, frac):
    ln_a = appro_ln_ref(a, w, frac)
    ln_b = appro_ln_ref(b, w, frac)
    log_ratio = clampW(ln_a - ln_b, w)
    if log_ratio > 0:
        exp_pos = appro_exp_ref(-log_ratio, w, frac)
        exp_pos = max(exp_pos, 1)  # avoid div-by-zero in recip_ref's own msb_pos(0) edge case
        return recip_ref(exp_pos, w, frac)
    else:
        return appro_exp_ref(log_ratio, w, frac)


def vectors(w, frac, n, rng):
    maxv = (1 << (w - 1)) - 1
    scale = 1 << frac
    min_v = max(1, scale // 8)
    v = [(rng.randint(min_v, maxv), rng.randint(min_v, maxv)) for _ in range(n)]
    v += [(scale, scale), (scale, 2 * scale if 2 * scale <= maxv else scale), (2 * scale if 2 * scale <= maxv else scale, scale)]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {' '.join(RTL)}", f"chparam -set W {w} -set FRAC {frac} quark_shared_div",
             "hierarchy -top quark_shared_div", "proc -norom", "flatten", "select quark_shared_div"]
    for a, b in vecs:
        lines.append(f"eval -set a {w}'d{a & ((1 << w) - 1)} -set b {w}'d{b & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(113)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 100, rng)
        got = run(w, frac, vecs)
        exp = [ref(a, b, w, frac) for a, b in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_shared_div W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   a,b =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
