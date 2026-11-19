"""Synthesises argmax_tree at each swept lane count and writes
results/argmax.json. W is fixed at 32 (Q12.20's word width); only N (lane
count) is swept. k=1 only (plain max/argmax) -- no top-k variant."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "argmax_tree.v")
W = 32
LANE_COUNTS = [4, 8, 16, 32, 64]


def main():
    rows = []
    for n in LANE_COUNTS:
        res = synth.synth([RTL], "argmax_tree", {"N": n, "W": W})
        row = report.row("argmax_tree", {"N": n, "W": W, "k": 1, "in": "Q12.20", "out": "Q12.20 + index"},
                          "comparator tree", res,
                          "taps both max value (cim.max) and argmax index (cim.argmax) from one circuit")
        rows.append(row)
        print(f"  argmax_tree N={n}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "argmax", rows)


if __name__ == "__main__":
    main()
