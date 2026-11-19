"""Functional check of e2softmax_row8.v against a Python model of the same
fixed-point arithmetic (SOLE's Eq.7-13). Yosys eval on random and edge-case score
rows. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "e2softmax_row8.v")
BASE_RTL = os.path.join(HERE, "rtl", "softmax_row8.v")  # for max_reduce8


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def log2exp_fixed(d):
    x = d << 8
    xshift = x + (x >> 1) - (x >> 4)
    k_raw = xshift >> 8
    k = -k_raw
    return max(0, min(15, k))


def aldivision_fixed(k_y, s_sum):
    p_bit = s_sum.bit_length() - 1
    k_s = p_bit - 8
    next_bit = 0 if p_bit == 0 else (s_sum >> (p_bit - 1)) & 1
    const_hi, const_lo = 419, 291
    const = const_lo if next_bit else const_hi
    ks_sum = k_y + k_s + 1
    if ks_sum >= 0:
        return 0 if ks_sum >= 16 else to_signed(const >> ks_sum, 16)
    else:
        return 32767 if ks_sum <= -16 else to_signed((const << (-ks_sum)) & 0xFFFF, 16)


def ref_row(scores):
    max_val = max(scores)
    ds = [s - max_val for s in scores]
    kys = [log2exp_fixed(d) for d in ds]
    es = [0 if k > 8 else (256 >> k) for k in kys]
    s_sum = sum(es)
    return [aldivision_fixed(ky, s_sum) for ky in kys]


def vectors(n, rng):
    rows = [[rng.randint(0, 255) for _ in range(8)] for _ in range(n)]
    rows += [[0] * 8, [255] * 8, [255, 0, 0, 0, 0, 0, 0, 0], [0, 1, 2, 3, 4, 5, 6, 7], [255, 254, 1, 0, 0, 0, 0, 0]]
    return rows


def run(rows):
    lines = [f"read_verilog -sv {BASE_RTL}", f"read_verilog -sv {RTL}", "hierarchy -top e2softmax_row8",
             "proc -norom", "flatten", "select e2softmax_row8"]
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
    rng = random.Random(97)
    rows = vectors(60, rng)
    got = run(rows)
    exp = [ref_row(row) for row in rows]
    bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
    status = "OK" if len(got) == len(rows) and not bad else "FAIL"
    print(f"e2softmax_row8  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  {status}")
    for r, g, e in bad[:5]:
        print("   scores =", r, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
