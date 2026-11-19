"""Mapping latency figure, RSE list scheduling: every app at the greedy sweep's caps
(run_scripts/rse_listsched_sweep.py; needs mapping_rse_greedy.py first), checked exactly
against the figure; writes reproduce/results/mapping_rse_listsched.csv. Skipped if its
results are complete; --force deletes them and reruns. See mapping_common.py.

Usage (from the repo root): python3 reproduce/mapping_rse_listsched.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_common import run_method  # noqa: E402

if __name__ == "__main__":
    run_method("rse", "listsched", __doc__)
