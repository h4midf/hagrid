"""Functional check of ibert_softmax_row8.v against a Python model of the same
fixed-point arithmetic (I-BERT's Eq.10-14 exp and softmax_row8.v's normalize). Yosys
eval on random and edge-case score rows. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "ibert_softmax_row8.v")
BASE_RTL = os.path.join(HERE, "rtl", "softmax_row8.v")  # for max_reduce8

LN2_Q88, INV_LN2_Q88 = 177, 369
C0, C1, C2 = 92, 346, 88
N_MAX = 9


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def i_exp(d):
    """d: plain 8-bit score diff (<=0). Mirrors ibert_exp_pwl.v exactly,
    including its width-tightened p/term containers (masked to their
    declared bit widths, matching the RTL's own truncation)."""
    x = d << 8
    negx = -x
    zq_full = negx * INV_LN2_Q88
    z = zq_full >> 16
    lnz = LN2_Q88 * z
    p = x + lnz
    p_t = to_signed(p & 0xFFF, 12)  # TW=12 in the RTL
    term = p_t + C1
    sq = (term * term + 128) >> 8
    sq = to_signed(sq & 0xFFF, 12)
    c0sq = (C0 * sq + 128) >> 8
    exp_p = c0sq + C2
    if z >= N_MAX:
        return 0
    return exp_p >> (z & 0xF)


def ref_row(scores):
    max_val = max(scores)
    ds = [s - max_val for s in scores]
    es = [i_exp(d) for d in ds]
    exp_sum = sum(es)
    recip = 0 if exp_sum == 0 else to_signed((65536 // exp_sum) & 0xFFFF, 16)
    return [to_signed((e * recip) >> 8, 16) for e in es]


def vectors(n, rng):
    rows = [[rng.randint(0, 255) for _ in range(8)] for _ in range(n)]
    rows += [[0] * 8, [255] * 8, [255, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7]]
    return rows


def run(rows):
    lines = [f"read_verilog -sv {BASE_RTL}", f"read_verilog -sv {RTL}", "hierarchy -top ibert_softmax_row8",
             "proc -norom", "flatten", "select ibert_softmax_row8"]
    for row in rows:
        sets = " ".join(f"-set s{i} 8'd{row[i]}" for i in range(8))
        shows = " ".join(f"-show p{i}" for i in range(8))
        lines.append(f"eval {sets} {shows}")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    cur = {}
    for name, raw in re.findall(r"Eval result: \\(p\d) = ([^\n]+)\.", out):
        m = re.match(r"16'([01]+)$", raw)
        cur[name] = to_signed(int(m.group(1), 2), 16) if m else int(raw)
        if len(cur) == 8:
            results.append([cur[f"p{i}"] for i in range(8)])
            cur = {}
    return results


def main():
    rng = random.Random(71)
    rows = vectors(60, rng)
    got = run(rows)
    exp = [ref_row(row) for row in rows]
    bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
    status = "OK" if len(got) == len(rows) and not bad else "FAIL"
    print(f"ibert_softmax_row8  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  {status}")
    for r, g, e in bad[:5]:
        print("   scores =", r, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
