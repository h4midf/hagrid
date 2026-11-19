"""Draws the paper's mapping latency figure (figures/evaluation/mapping_latency_comparison.tex)
from the reproduced results: plot/mapping_latency.py --extract reads the greedy, list-
scheduling and CP-SAT results (run_scripts/results/) and both GPU files (dump/gpu_baseline/),
rewrites the figure's saved data (plot/meta_data/mapping_latency.json) and the .tex.

Run after its inputs exist, each from its own script, per target (fefet1c, hyflexpim, rse):
  greedy             reproduce/mapping_<target>_greedy.py
  list scheduling    reproduce/mapping_<target>_listsched.py (after greedy)
  CP-SAT             reproduce/mapping_<target>_cpsat.py
  GPU lines          reproduce/gpu_baseline.py, on each GPU machine

Raises if an input is missing. Then lists every value of the figure that changed from the
saved data it replaced (greedy and list scheduling should not; CP-SAT and the GPUs vary
from run to run).

Usage (from the repo root):
  python3 reproduce/mapping_latency.py
"""
import json
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO  # noqa: E402
from mapping_common import FIGURE_DATA, MODELS, TARGETS, figure_rows, summary_path  # noqa: E402

GPU_FILES = {"--gpu": REPO / "dump" / "gpu_baseline" / "latency_rtx3090.json",
             "--gpu-mobile": REPO / "dump" / "gpu_baseline" / "latency_rtx3060_laptop.json"}
METHODS = ("greedy", "listsched", "cpsat", "gpu", "gpu_mobile")


def check_inputs() -> None:
    missing = [p for p in GPU_FILES.values() if not p.exists()]
    missing += [summary_path(t, m, model) for t in TARGETS for m in ("greedy", "listsched", "cpsat")
                for model in MODELS if not summary_path(t, m, model).exists()]
    if missing:
        raise SystemExit("missing inputs (run their reproduce/ scripts first):\n  "
                         + "\n  ".join(str(p.relative_to(REPO)) for p in missing))


def main() -> None:
    check_inputs()
    before = figure_rows()
    args = [a for flag, path in GPU_FILES.items() for a in (flag, str(path))]
    subprocess.run([sys.executable, str(REPO / "plot" / "mapping_latency.py"), "--extract", *args],
                   cwd=REPO, check=True)
    after = figure_rows()
    changed = 0
    for target in TARGETS:
        for model in MODELS:
            for old, new in zip(before[target][model], after[target][model]):
                for method in METHODS:
                    if old[method]["latency_ns"] != new[method]["latency_ns"]:
                        changed += 1
                        print(f"  changed {target} {model} cap {new['cap']} {method}: "
                              f"{old[method]['latency_ns']} -> {new[method]['latency_ns']}")
    print(f"rewrote {FIGURE_DATA.relative_to(REPO)} and the figure; {changed} values changed")


if __name__ == "__main__":
    main()
