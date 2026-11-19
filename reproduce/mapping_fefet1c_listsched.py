"""Mapping latency figure, 1F1C list scheduling: every model at the greedy sweep's caps
(run_scripts/fefet1c_listsched_sweep.py; needs mapping_fefet1c_greedy.py first), checked
exactly against the figure; writes reproduce/results/mapping_fefet1c_listsched.csv.
Skipped if its results are complete; --force deletes them and reruns. See mapping_common.py.

Usage (from the repo root): python3 reproduce/mapping_fefet1c_listsched.py [--force]
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mapping_common import run_method  # noqa: E402

if __name__ == "__main__":
    run_method("fefet1c", "listsched", __doc__)
