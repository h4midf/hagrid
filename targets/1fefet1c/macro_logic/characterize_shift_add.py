"""Characterizes one 1F1C array's shift-and-add for the config (F1C_CONFIG) and writes
<config>/macro_logic/results/shift_add.json:
  - the UNITS shared datapaths (rtl/shift_add.v, or rtl/shift_add_flip.v with the flip
    correction and flip bits when design.json has ISAAC's weight-flip encoding) with
    the shared yosys / Nangate45 flow: area at 45 nm and rescaled to 28 nm, leakage,
    dynamic energy per cycle
  - the accumulators: UNITS SRAM banks of OUTS / UNITS words of ACC bits, with NVSim
    at 28 nm (<config>/macro_logic/acc_bank.cfg): area, read / write energy, leakage
    per bank
Every slot, each datapath reads one word of its bank and writes it back.

Usage (from the repo root):
  F1C_CONFIG=128 python3 targets/1fefet1c/macro_logic/characterize_shift_add.py
"""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
import report  # noqa: E402
import synth  # noqa: E402
from params import ACC, CELLS, CODE, CV, ENCODED, OUT_DIR, OUTS, PART, UNITS  # noqa: E402
from run_nvsim import NVSIM, grab, grab_unit  # noqa: E402

MODULE = "shift_add"
SRAM_CFG = "acc_bank.cfg"


def acc_bank() -> dict:
    """NVSim of one accumulator bank; fails if acc_bank.cfg does not match params.py."""
    words = OUTS // UNITS
    if (words * ACC) % 8:
        raise ValueError(f"a bank of {words} x {ACC} bits is not a whole number of bytes")
    cfg = open(os.path.join(OUT_DIR, SRAM_CFG)).read()
    want = {"Capacity (B)": words * ACC // 8, "WordWidth (bit)": ACC}
    for key, value in want.items():
        got = int(grab(rf"^-{re.escape(key)}: (\d+)", cfg))
        if got != value:
            raise ValueError(f"{OUT_DIR}/{SRAM_CFG} has {key} {got}, the shift-and-add needs {value}")
    node = int(grab(r"^-ProcessNode: (\d+)", cfg))
    out = subprocess.run([str(NVSIM), SRAM_CFG], cwd=OUT_DIR, capture_output=True, text=True, check=True).stdout
    return {
        "banks": UNITS, "words": words, "word_bits": ACC, "node_nm": node,
        "area_um2": grab(r"^ - Total Area = [\d.]+um x [\d.]+um = ([\d.]+)um\^2", out),
        "read_pJ": grab_unit("Read Dynamic Energy", "J", out, "p"),
        "write_pJ": grab_unit("Write Dynamic Energy", "J", out, "p"),
        "read_latency_ps": grab_unit("Read Latency", "s", out, "p"),
        "leakage_uW": grab_unit("Leakage Power", "W", out, "n") / 1000,
        "source": f"NVSim, {SRAM_CFG} (NVSim's 6T SRAM.cell)",
    }


if __name__ == "__main__":
    if ENCODED:
        rtl = "shift_add_flip"
        params = {"OUTS": OUTS, "CELLS": CELLS, "CODE": CODE, "CV": CV, "PART": PART, "ACC": ACC, "UNITS": UNITS}
        method = "shift-and-add, bit-serial input, shared datapaths, ISAAC weight-flip correction and flip bits"
    else:
        rtl = "shift_add"
        params = {"OUTS": OUTS, "CELLS": CELLS, "CODE": CODE, "PART": PART, "ACC": ACC, "UNITS": UNITS}
        method = "shift-and-add, bit-serial input, shared datapaths"
    res = synth.synth([os.path.join(HERE, "rtl", rtl + ".v")], rtl, params)
    row = report.row(MODULE, params, method, res,
                     f"one 1F1C array (rtl/{rtl}.v): int8 weights over 2-bit cells, 8-bit inputs MSB first; "
                     "area/leakage/dynamic are the datapaths only, the accumulators are acc_sram")
    row["rtl"] = rtl
    row["acc_sram"] = acc_bank()
    report.save(OUT_DIR, MODULE, [row])
    s = row["acc_sram"]
    print(f"  {MODULE} ({rtl}): datapaths {res['area_um2']:.1f} um^2 (45 nm), {row['area_um2_28nm']} um^2 (28 nm); "
          f"accumulators {s['banks']} x {s['area_um2']:.1f} um^2 ({s['node_nm']} nm)", flush=True)
