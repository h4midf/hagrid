"""Functional check of elu.v (and its exp_pwl_q12_20 / exp_pwl_q8_8 primitives)
against a Python model of the same fixed-point arithmetic, bit for bit, not against
true ELU (its error vs. true ELU: approximation_error.py).

Each vector is evaluated by Yosys (eval) on random and edge-case inputs,
at both characterized widths (W=32/FRAC=20/Q12.20 and W=16/FRAC=8/Q8.8).
Exit code 0 only if every vector matches, for every swept (width, alpha).
"""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", "exp_pwl_q12_20.v"), os.path.join(HERE, "rtl", "exp_pwl_q8_8.v"),
       os.path.join(HERE, "rtl", "elu.v")]

TABLES = {
    32: dict(FRAC=20, LOG2E_Q=1512775, N_MAX=21,
             OFFSET=[1048576, 1004120, 961548, 920782, 881744, 844361, 808563, 774282,
                     741455, 710020, 679917, 651091, 623487, 597053, 571740, 547500],
             SLOPE=[-44456, -42572, -40766, -39038, -37383, -35798, -34281, -32827,
                    -31435, -30103, -28826, -27604, -26434, -25313, -24240, -23212]),
    16: dict(FRAC=8, LOG2E_Q=369, N_MAX=9,
             OFFSET=[256, 245, 235, 225, 215, 206, 197, 189, 181, 173, 166, 159, 152, 146, 140, 134],
             SLOPE=[-11, -10, -10, -10, -9, -9, -8, -8, -8, -7, -7, -7, -6, -6, -6, -6]),
}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clamp(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def exp_pwl(x, w):
    t = TABLES[w]
    frac, scale = t["FRAC"], 1 << t["FRAC"]
    y_full = x * t["LOG2E_Q"]
    y = to_signed(y_full >> frac, 64)
    neg_y = -y
    n_raw = neg_y >> frac
    r_frac = neg_y & (scale - 1)
    seg = r_frac >> (frac - 4)
    intra = r_frac & ((1 << (frac - 4)) - 1)
    off, slope = t["OFFSET"][seg], t["SLOPE"][seg]
    g_full = off + (slope * intra) // (1 << (frac - 4))
    if n_raw >= t["N_MAX"]:
        return 0
    return g_full >> n_raw


def ref(x, alpha_q, w):
    frac, scale = TABLES[w]["FRAC"], 1 << TABLES[w]["FRAC"]
    if x > 0:
        return x
    e = exp_pwl(x, w)
    term = e - scale
    half = 1 << (frac - 1)
    p = term * alpha_q
    return clamp((p + half) >> frac, w)


def vectors(w, n, rng):
    scale = 1 << TABLES[w]["FRAC"]
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    lo, hi = max(minv, -20 * scale), min(maxv, 20 * scale)
    v = [rng.randint(lo, hi) for _ in range(n)]
    v += [0, 1, -1, maxv, minv, -scale, -2 * scale if -2 * scale >= minv else minv]
    return v


def run(w, frac, alpha_q, vecs):
    lines = [f"read_verilog -sv {' '.join(RTL)}", f"chparam -set W {w} -set FRAC {frac} -set ALPHA "
              f"{alpha_q & ((1<<w)-1)} elu", "hierarchy -top elu", "proc -norom", "flatten", "select elu"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(31)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        scale = 1 << frac
        for alpha_name, alpha_q in (("1.0", scale), ("0.5", scale // 2)):
            vecs = vectors(w, 150, rng)
            got = run(w, frac, alpha_q, vecs)
            exp = [ref(x, alpha_q, w) for x in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"elu W={w:2d} alpha={alpha_name:4s}  evaluated {len(got)}/{len(vecs)}  "
                  f"mismatches {len(bad)}  {status}")
            for v, g, e in bad[:5]:
                print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
