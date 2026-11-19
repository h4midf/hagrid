"""Synthesizes vec_select at every swept lane count and data width and writes
results/vec_select.json. W: 32 = Q12.20, 16 = Q8.8 (the two data widths). The lane
counts are vec_add's.

Usage (from the repo root):
  python3 custom_op_library/vec_select/characterize.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "vec_select.v")
LANE_COUNTS = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
FORMATS = {32: "Q12.20", 16: "Q8.8"}


def main():
    rows = []
    for w, fmt in FORMATS.items():
        for n in LANE_COUNTS:
            res = synth.synth([RTL], "vec_select", {"N": n, "W": w})
            rows.append(report.row(
                "vec_select", {"N": n, "W": w, "in": fmt, "out": fmt},
                "2-to-1 multiplexer per lane", res, "out = mask ? x : fill (shared fill word)"))
            print(f"  vec_select N={n} W={w}: area {res['area_um2']:.0f} um^2 (45 nm), "
                  f"delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "vec_select", rows)


if __name__ == "__main__":
    main()
