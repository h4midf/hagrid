"""Functional check of absmax_reduce_tree.v against token_quant_ref.absmax (exact: no
rounding). Yosys eval at N=8 (small sanity check) and N=256 (the reduction width the
softmax pipeline uses; a 768- or 3072-wide row is 3 or 12 passes). Exit code 0 only if
every vector matches."""
import os
import random
import re
import subprocess
import sys

from token_quant_ref import absmax

sys.set_int_max_str_digits(100000)  # N=256 packs an 8192-bit flat vector into one decimal literal

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "absmax_reduce_tree.v")


def pack(vals, w):
    flat = 0
    for i, v in enumerate(vals):
        flat |= (v & ((1 << w) - 1)) << (i * w)
    return flat


def run(n, w, rows):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set N {n} -set W {w} absmax_reduce_tree",
             "hierarchy -top absmax_reduce_tree", "proc -norom", "flatten", "select absmax_reduce_tree"]
    for row in rows:
        lines.append(f"eval -set x_flat {n * w}'d{pack(row, w)} -show absmax")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\absmax = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        results.append(int(m.group(2), 2) if m else int(raw))
    return results


def vectors(n, count, w, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    rows = [[rng.randint(minv, maxv) for _ in range(n)] for _ in range(count)]
    # small magnitudes (activation-like) and edge cases
    rows += [[rng.randint(-(1 << 20), 1 << 20) for _ in range(n)] for _ in range(count)]
    rows += [[0] * n, [maxv] * n, [minv] * n, [-1] * n]
    rows[0][n // 2] = minv          # |-2^(W-1)| = 2^(W-1), the largest magnitude
    rows[1][0] = maxv
    rows[2] = [-5 if i == n - 1 else 3 for i in range(n)]   # negative lane wins
    return rows


def main():
    rng = random.Random(13)
    status = "OK"
    for n in (8, 256):
        rows = vectors(n, 15, 32, rng)
        got = run(n, 32, rows)
        exp = [absmax(row) for row in rows]
        bad = [(g, e) for g, e in zip(got, exp) if g != e]
        ok = len(got) == len(rows) and not bad
        print(f"absmax_reduce_tree N={n} W=32  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            status = "FAIL"
            for g, e in bad[:3]:
                print("   mismatch: rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
