"""Hardware validation table, (a) 1F1C against Laleni et al. (JXCDC 2024): compiles NVSim,
runs NVSim, the column SPICE and the SAR ADC ENOB SPICE, computes the macro TOPS/W, and
writes reproduce/results/hardware_validation_1f1c.csv checked against the paper.
A step whose output exists is skipped; --force deletes the results and reruns everything.
See hardware_validation.py.

Usage (from the repo root): python3 reproduce/hardware_validation_1f1c.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from hardware_validation import run  # noqa: E402

if __name__ == "__main__":
    run("1f1c", __doc__)
