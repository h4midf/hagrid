"""Mapping latency figure, HyFlexPIM CP-SAT: every model at three array caps, one hour per
case (run_scripts/hyflexpim_cpsat_sweep.py; up to 24 hours), compared with the figure
(CP-SAT is not deterministic: relative differences); writes
reproduce/results/mapping_hyflexpim_cpsat.csv. Skipped if its results are complete;
--force deletes them and its mappings, and reruns. See mapping_common.py.

Usage (from the repo root): python3 reproduce/mapping_hyflexpim_cpsat.py --cpsat-workers 24 [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_common import run_method  # noqa: E402

if __name__ == "__main__":
    run_method("hyflexpim", "cpsat", __doc__)
