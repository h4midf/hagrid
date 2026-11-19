"""Synthesizes rtl/rram_shift_add.v with the shared yosys / NanGate45 flow and writes
results/rram_shift_add.json: one unit at 45 nm and at the chip's 65 nm, 100 MHz, and the
macro's units (one per read-out circuit).

Usage (from the repo root):
  python3 targets/rram-sram-ensemble/periphery/characterize_rram_shift_add.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
sys.path.insert(0, HERE)
import report  # noqa: E402
import synth  # noqa: E402
from scaling import DESIGN, at_chip  # noqa: E402


def main() -> None:
    res = synth.synth([os.path.join(HERE, "rtl", "rram_shift_add.v")], "rram_shift_add", {})
    row = report.row("rram_shift_add", {"ACC": 9}, "shift-and-add of one read-out circuit", res, "combinational")
    unit = at_chip(row)
    units = DESIGN["rram_macro"]["readout"]["readout_circuits"]
    row["chip_65nm"] = {"unit": unit, "units": units, "area_um2": unit["area_um2"] * units,
                        "power_mW": unit["power_mW"] * units}
    report.save(HERE, "rram_shift_add", [row])
    print(f"rram_shift_add: {res['area_um2']:.0f} um^2 (45 nm) -> {unit['area_um2']:.0f} um^2, "
          f"{unit['power_mW']:.4f} mW; x{units} = {row['chip_65nm']['power_mW']:.4f} mW")


if __name__ == "__main__":
    main()
