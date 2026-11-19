"""Synthesizes rtl/rram_decoder.v with the shared yosys / NanGate45 flow and writes
results/rram_decoder.json: the circuit at 45 nm and at the chip's 65 nm, 100 MHz.

Usage (from the repo root):
  python3 targets/rram-sram-ensemble/periphery/characterize_rram_decoder.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
sys.path.insert(0, HERE)
import report  # noqa: E402
import synth  # noqa: E402
from scaling import at_chip  # noqa: E402


def main() -> None:
    res = synth.synth([os.path.join(HERE, "rtl", "rram_decoder.v")], "rram_decoder", {})
    row = report.row("rram_decoder", {}, "RRAM decoder: FSM, column counter, one-hot programming selects", res)
    row["chip_65nm"] = at_chip(row)
    report.save(HERE, "rram_decoder", [row])
    c = row["chip_65nm"]
    print(f"rram_decoder: {res['area_um2']:.0f} um^2 (45 nm) -> {c['area_um2']:.0f} um^2, {c['power_mW']:.4f} mW "
          f"({c['dynamic_mW']:.4f} dynamic + {c['leakage_mW']:.4f} leakage)")


if __name__ == "__main__":
    main()
