"""Functional check of rms_norm.v against a Python model of the same fixed-point
arithmetic (bit for bit). Yosys eval on random and edge-case N-lane rows, at both
widths. eps is always > 0, as in RMSNorm. Exit code 0 only if every vector matches,
for every (width, N).
"""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", "reduce_tree.v"), os.path.join(HERE, "rtl", "rsqrt_q12_20.v"),
       os.path.join(HERE, "rtl", "rsqrt_q8_8.v"), os.path.join(HERE, "rtl", "rms_norm.v")]
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
    table = {}
    for parity in (0, 1):
        for idx in range(1 << IDX_BITS):
            m = 1.0 + idx / (1 << IDX_BITS)
            f = (1.0 / math.sqrt(m)) if parity == 0 else (1.0 / math.sqrt(2 * m))
            table[(parity, idx)] = round(f * scale)
    return table


SEEDS = {w: build_seed_table(1 << cfg["frac"]) for w, cfg in WIDTH_CFG.items()}


def rsqrt_fixed(val, w):
    frac = WIDTH_CFG[w]["frac"]
    scale = 1 << frac
    one_half_q = round(1.5 * scale)
    p = val.bit_length() - 1
    p_real = p - frac
    e = p_real // 2
    parity = p_real & 1
    if p >= IDX_BITS:
        idx = (val >> (p - IDX_BITS)) & ((1 << IDX_BITS) - 1)
    else:
        idx = (val << (IDX_BITS - p)) & ((1 << IDX_BITS) - 1)
    seed = SEEDS[w][(parity, idx)]
    y = (seed >> e) if e >= 0 else (seed << (-e))
    yy_q = (y * y) >> frac
    vyy_q = (val * yy_q) >> frac
    term_q = one_half_q - (vyy_q >> 1)
    y = (y * term_q) >> frac
    return clampW(y, w)


def ref(lanes, gamma, eps, w):
    frac = WIDTH_CFG[w]["frac"]
    n = len(lanes)
    levels = int(math.log2(n))
    half = 1 << (frac - 1)

    sqs = [clampW((x * x + half) >> frac, w) for x in lanes]
    ms = to_signed((sum(sqs) >> levels) & ((1 << w) - 1), w)

    if ms == 0:
        s = 0
    else:
        r1 = rsqrt_fixed(ms, w)
        s = clampW((ms * r1 + half) >> frac, w)

    s_eps = clampW(s + eps, w)
    v2 = clampW((s_eps * s_eps + half) >> frac, w)
    inv = rsqrt_fixed(v2, w) if v2 > 0 else 0

    norms = [clampW((x * inv + half) >> frac, w) for x in lanes]
    return [clampW((nv * gamma + half) >> frac, w) for nv in norms]


def pack_flat(lanes, w):
    v = 0
    for i, lane in enumerate(lanes):
        v |= (lane & ((1 << w) - 1)) << (i * w)
    return v


def unpack_flat(v, n_lanes, w, bits):
    v &= (1 << bits) - 1
    return tuple(to_signed((v >> (i * w)) & ((1 << w) - 1), w) for i in range(n_lanes))


def parse_eval(out, signal):
    got = []
    for raw in re.findall(rf"Eval result: \\{signal} = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        got.append(int(m.group(2), 2) if m else int(raw))
    return got


def vectors(n_lanes, w, n, rng):
    frac = WIDTH_CFG[w]["frac"]
    scale = 1 << frac
    v = []
    for _ in range(n):
        v.append((tuple(rng.randint(-10 * scale, 10 * scale) for _ in range(n_lanes)),
                   rng.randint(scale // 4, 2 * scale), rng.randint(1, scale // 4)))
    v.append(((0,) * n_lanes, scale, 1))                          # all-zero -> ms=0 guard path
    v.append(((0,) * n_lanes, scale, scale // 2))                  # all-zero, larger eps
    v.append((tuple(range(1, n_lanes + 1)), scale, 1))             # ramp
    v.append(((scale,) * n_lanes, scale, 1))                       # all-tied, nonzero ms
    return v


def run(n_lanes, w, vecs):
    frac = WIDTH_CFG[w]["frac"]
    lines = [f"read_verilog -sv {' '.join(RTL)}", f"chparam -set N {n_lanes} -set W {w} -set FRAC {frac} "
              "rms_norm", "hierarchy -top rms_norm", "proc -norom", "flatten", "select rms_norm"]
    for lanes, gamma, eps in vecs:
        lines.append(f"eval -set x_flat {n_lanes*w}'d{pack_flat(lanes, w)} -set gamma {w}'d{gamma & ((1<<w)-1)} "
                      f"-set eps {w}'d{eps & ((1<<w)-1)} -show y_flat")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    return [unpack_flat(v, n_lanes, w, n_lanes * w) for v in parse_eval(out, "y_flat")]


def main():
    rng = random.Random(83)
    ok_all = True
    for w in (32, 16):
        for n_lanes in (4, 8, 16, 32, 64):
            vecs = vectors(n_lanes, w, 30, rng)
            got = run(n_lanes, w, vecs)
            exp = [tuple(ref(lanes, gamma, eps, w)) for lanes, gamma, eps in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"rms_norm W={w:2d} N={n_lanes:3d}  evaluated {len(got)}/{len(vecs)}  "
                  f"mismatches {len(bad)}  {status}")
            for v, g, e in bad[:3]:
                print("  ", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
