"""Functional check of aldivision_standalone.v against e2softmax_row8.v's
aldivision_q8_8 model, at SUM_W=20 and SUM_W=24. Exit code 0 only if every vector
matches."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "aldivision_standalone.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref_aldivision(k_y, s_sum):
    """Already-validated reference, see test_e2softmax_row8.py's own copy."""
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


def run(sum_w, rows):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set SUM_W {sum_w} aldivision_standalone",
              "hierarchy -top aldivision_standalone", "proc -norom", "flatten", "select aldivision_standalone"]
    for k_y, s_sum in rows:
        lines.append(f"eval -set k_y 4'd{k_y} -set s_sum {sum_w}'d{s_sum} -show p_out")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\p_out = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        results.append(to_signed(int(m.group(2), 2), int(m.group(1))) if m else to_signed(int(raw), 16))
    return results


def main():
    rng = random.Random(31)
    status = "OK"
    for sum_w, max_sum in ((20, 2048), (24, 131072)):
        rows = [(rng.randint(0, 15), rng.randint(1, max_sum)) for _ in range(100)]
        rows += [(0, 1), (15, max_sum), (8, max_sum // 2)]
        got = run(sum_w, rows)
        exp = [ref_aldivision(k, s) for k, s in rows]
        bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
        ok = len(got) == len(rows) and not bad
        print(f"aldivision_standalone SUM_W={sum_w}  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            status = "FAIL"
            for r, g, e in bad[:5]:
                print("   (k_y,s_sum) =", r, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
