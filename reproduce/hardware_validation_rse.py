"""Hardware validation table, (c) RSE against Krishnan et al. (TCAD 2022) Sec. V-F:
compiles CACTI, runs the logic, readout, WL and SRAM-array SPICE, the decoder, shift-add
and SRAM-macro synthesis, the macros' area and power and SRAM over RRAM, and writes
reproduce/results/hardware_validation_rse.csv checked against the paper.
A step whose output exists is skipped; --force deletes the results and reruns everything.
See hardware_validation.py.

Usage (from the repo root): python3 reproduce/hardware_validation_rse.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_validation import run  # noqa: E402

if __name__ == "__main__":
    run("rse", __doc__)
