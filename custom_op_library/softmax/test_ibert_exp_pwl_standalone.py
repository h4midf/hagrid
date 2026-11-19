"""Functional check of ibert_exp_pwl_standalone.v(d << 8) against
ibert_softmax_row8.v's ibert_exp_pwl model, for every d in -255..0. Exit code 0 only
if every vector matches."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "ibert_exp_pwl_standalone.v")

LN2_Q88, INV_LN2_Q88 = 177, 369
C0, C1, C2 = 92, 346, 88
N_MAX = 9


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def i_exp_original(d):
    """Already-validated reference, see test_ibert_softmax_row8.py's own copy."""
    x = d << 8
    negx = -x
    zq_full = negx * INV_LN2_Q88
    z = zq_full >> 16
    lnz = LN2_Q88 * z
    p = x + lnz
    p_t = to_signed(p & 0xFFF, 12)
    term = p_t + C1
    sq = (term * term + 128) >> 8
    sq = to_signed(sq & 0xFFF, 12)
    c0sq = (C0 * sq + 128) >> 8
    exp_p = c0sq + C2
    if z >= N_MAX:
        return 0
    return exp_p >> (z & 0xF)


def run(xs):
    lines = [f"read_verilog -sv {RTL}", "hierarchy -top ibert_exp_pwl_standalone", "proc -norom", "flatten",
             "select ibert_exp_pwl_standalone"]
    for x in xs:
        lines.append(f"eval -set x 24'd{x & 0xFFFFFF} -show exp_out")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\exp_out = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        results.append(to_signed(int(m.group(2), 2), int(m.group(1))) if m else int(raw))
    return results


def main():
    ds = list(range(-255, 1))
    xs = [d << 8 for d in ds]
    got = run(xs)
    exp = [i_exp_original(d) for d in ds]
    bad = [(d, g, e) for d, g, e in zip(ds, got, exp) if g != e]
    status = "OK" if len(got) == len(ds) and not bad else "FAIL"
    print(f"ibert_exp_pwl_standalone  evaluated {len(got)}/{len(ds)}  mismatches {len(bad)}  {status}")
    for d, g, e in bad[:5]:
        print("   d =", d, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
