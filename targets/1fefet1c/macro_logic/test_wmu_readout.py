"""Functional check of rtl/wmu_readout.v: after a two-conversion WMU read, every value
on the output matches the Python model.

Model: column j stores level (j + SEED) mod 4; its read difference is that level's
smallest observed difference (LEVEL_DIFFS), then, in a second pass, its largest; the
decoded level is the number of thresholds the difference reaches and must be the stored
level; value k is the 2-bit levels of columns k * CELLS_PER_VALUE .. (k + 1) *
CELLS_PER_VALUE - 1, lowest column in the lowest bits.

Usage (from the repo root):
  F1C_CONFIG=64 python3 targets/1fefet1c/macro_logic/test_wmu_readout.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import CELLS_PER_VALUE, CODE, COLS, LEVEL_DIFFS, THRESHOLDS  # noqa: E402
from sim import final_value, simulate  # noqa: E402

SEEDS = (0, 1, 2, 3)


def ref(seed: int, sel: int, diffs: list[int]) -> int:
    value = 0
    for i in range(CELLS_PER_VALUE):
        j = sel * CELLS_PER_VALUE + i
        stored = (j + seed) % 4
        level = sum(diffs[stored] >= t for t in THRESHOLDS)
        if level != stored:
            raise ValueError(f"read difference {diffs[stored]} of level {stored} decodes to {level}")
        value |= level << (2 * i)
    return value


def main():
    bad = []
    edges = {"smallest": [d[0] for d in LEVEL_DIFFS], "largest": [d[-1] for d in LEVEL_DIFFS]}
    cases = [(edge, seed, sel) for edge in edges for seed in SEEDS for sel in range(COLS // CELLS_PER_VALUE)]
    for edge, seed, sel in cases:
        diffs = edges[edge]
        vcd = simulate(["wmu_readout", "wmu_readout_tb"], "wmu_readout_tb",
                       {"COLS": COLS, "CODE": CODE, "CELLS_PER_VALUE": CELLS_PER_VALUE,
                        "T1": THRESHOLDS[0], "T2": THRESHOLDS[1], "T3": THRESHOLDS[2],
                        "DIFF1": diffs[1], "DIFF2": diffs[2], "DIFF3": diffs[3],
                        "SEED": seed, "SEL": sel}, 1)
        got, exp = final_value(vcd, "value"), ref(seed, sel, diffs)
        if got != exp:
            bad.append((edge, seed, sel, got, exp))
    print(f"wmu_readout: {len(cases) - len(bad)}/{len(cases)} values match")
    for edge, seed, sel, got, exp in bad[:5]:
        print(f"  {edge} diffs, seed {seed} sel {sel}: rtl {got:#x} ref {exp:#x}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
