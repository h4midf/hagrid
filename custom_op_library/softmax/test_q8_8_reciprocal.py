"""Functional check of q8_8_reciprocal.v against 65536 // sum (test_softmax.py's
normalize), at SUM_W=20, enough for a row of 512 (max sum 131072). Exit code 0 only
if every vector matches."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "q8_8_reciprocal.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref_recip(s):
    return to_signed(0 if s == 0 else (65536 // s), 16)


def run(sum_w, sums):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set SUM_W {sum_w} q8_8_reciprocal",
              "hierarchy -top q8_8_reciprocal", "proc -norom", "flatten", "select q8_8_reciprocal"]
    for s in sums:
        lines.append(f"eval -set sum {sum_w}'d{s & ((1 << sum_w) - 1)} -show recip")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\recip = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        results.append(to_signed(int(m.group(2), 2), int(m.group(1))) if m else to_signed(int(raw), 16))
    return results


def main():
    status = "OK"
    for sum_w, sums in ((20, [256, 512, 2048, 1, 2047, 2 * 256, 8 * 256]),
                         (20, [131072, 65536, 32768, 256, 1024, 512 * 256])):
        got = run(sum_w, sums)
        exp = [ref_recip(s) for s in sums]
        bad = [(s, g, e) for s, g, e in zip(sums, got, exp) if g != e]
        ok = len(got) == len(sums) and not bad
        print(f"q8_8_reciprocal SUM_W={sum_w}  evaluated {len(got)}/{len(sums)}  mismatches {len(bad)}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            status = "FAIL"
            for s, g, e in bad[:5]:
                print("   sum =", s, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
