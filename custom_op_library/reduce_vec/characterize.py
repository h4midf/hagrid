"""Synthesizes reduce_tree at every swept input count and data width and writes
results/reduce_vec.json.

Each reduce op uses the smallest tree with at least as many inputs as the values it
sums (compiler/xdsl_accel_width_stamp.py); a larger reduction takes several passes
of the largest tree. W: 32 = Q12.20, 16 = Q8.8 (the two data widths).

Usage (from the repo root):
  python3 custom_op_library/reduce_vec/characterize.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "reduce_tree.v")
INPUT_COUNTS = (8, 16, 32, 64, 128, 256, 512, 1024)
FORMATS = {32: "Q12.20", 16: "Q8.8"}


def main():
    rows = []
    for w, fmt in FORMATS.items():
        for n in INPUT_COUNTS:
            res = synth.synth([RTL], "reduce_tree", {"N": n, "W": w})
            out_bits = w + int(math.log2(n))
            rows.append(report.row(
                "reduce_tree", {"N": n, "W": w, "in": fmt, "out": f"{out_bits}-bit extended"},
                "binary adder tree", res, "no saturation: the output keeps full carry-safe precision"))
            print(f"  reduce_tree N={n} W={w}: area {res['area_um2']:.0f} um^2 (45 nm), "
                  f"delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "reduce_vec", rows)


if __name__ == "__main__":
    main()
