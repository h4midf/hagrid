"""Synthesizes rtl/macro_ctrl.v (one 1F1C array's input latches, word-line gating and
phase sequencer) with the shared yosys / Nangate45 flow and writes
<config>/macro_logic/results/macro_ctrl.json (area at 45 nm and rescaled to 28 nm).

Usage (from the repo root):
  F1C_CONFIG=128 python3 targets/1fefet1c/macro_logic/characterize_macro_ctrl.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
import report  # noqa: E402
import synth  # noqa: E402
from params import CONV_CYCLES, OUT_DIR, ROWS  # noqa: E402

MODULE = "macro_ctrl"

if __name__ == "__main__":
    params = {"ROWS": ROWS, "CONV_CYCLES": CONV_CYCLES}
    res = synth.synth([os.path.join(HERE, "rtl", MODULE + ".v")], MODULE, params)
    row = report.row(MODULE, params, "input latches, word-line gating, phase sequencer", res,
                     "one 1F1C array per MAC; the word-line drivers are in the NVSim array area")
    report.save(OUT_DIR, MODULE, [row])
    print(f"  {MODULE}: {res['area_um2']:.1f} um^2 (45 nm), {row['area_um2_28nm']} um^2 (28 nm)", flush=True)
