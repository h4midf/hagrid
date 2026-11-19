"""Synthesizes rtl/wmu_readout.v (one 1F1C array's WMU readout: per-column reference
latch, value select, and level decode of the selected value only) with the shared yosys /
Nangate45 flow and writes <config>/macro_logic/results/wmu_readout.json (area at 45 nm
and rescaled to 28 nm).

Usage (from the repo root):
  F1C_CONFIG=128 python3 targets/1fefet1c/macro_logic/characterize_wmu_readout.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
import report  # noqa: E402
import synth  # noqa: E402
from params import CELLS_PER_VALUE, CODE, COLS, OUT_DIR, THRESHOLDS  # noqa: E402

MODULE = "wmu_readout"

if __name__ == "__main__":
    params = {"COLS": COLS, "CODE": CODE, "CELLS_PER_VALUE": CELLS_PER_VALUE,
              "T1": THRESHOLDS[0], "T2": THRESHOLDS[1], "T3": THRESHOLDS[2]}
    res = synth.synth([os.path.join(HERE, "rtl", MODULE + ".v")], MODULE, params)
    row = report.row(MODULE, params, "two-conversion read: reference latches, value select, shared level decode", res,
                     "one 1F1C array; thresholds halfway between the read differences of results/mac.json cap_read")
    report.save(OUT_DIR, MODULE, [row])
    print(f"  {MODULE}: {res['area_um2']:.1f} um^2 (45 nm), {row['area_um2_28nm']} um^2 (28 nm)", flush=True)
