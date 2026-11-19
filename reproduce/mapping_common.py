"""What the mapping latency reproduction scripts share: the figure's models and targets
(from plot/mapping_latency.py, the script that draws the figure), each target's sweep
scripts and result files, the figure's saved data (plot/meta_data/mapping_latency.json,
the values the paper's figure is drawn from), and run_method, which every
<target>_<method>.py script calls.
"""
import argparse
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from checkpoints import CHECKPOINTS, checkpoint_steps
from common import REPO, RESULTS, Step, run_steps, write_csv

PY = sys.executable
CPSAT_TIME_LIMIT_S = 3600   # per case: one model at one cap (the paper: one hour)

RESULTS_DIR = REPO / "run_scripts" / "results"
FIGURE_DATA = REPO / "plot" / "meta_data" / "mapping_latency.json"


def _plot_module():
    sys.path.insert(0, str(REPO / "plot"))
    spec = importlib.util.spec_from_file_location("plot_mapping_latency", REPO / "plot" / "mapping_latency.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


PLOT = _plot_module()
MODELS = PLOT.MODELS
TARGETS = [key for key, _ in PLOT.TARGETS]


@dataclass(frozen=True)
class TargetSweeps:
    """One target's sweep scripts (modules under run_scripts) and their model argument."""
    greedy: str
    listsched: str
    cpsat: str
    models_flag: str        # --models, or --apps for the RRAM/SRAM ensemble
    cpsat_extra: list[str]  # further CP-SAT sweep arguments


SWEEPS = {
    "fefet1c": TargetSweeps("run_scripts.fefet1c_greedy_sweep", "run_scripts.fefet1c_listsched_sweep",
                            "run_scripts.fefet1c_cpsat_sweep", "--models", []),
    "hyflexpim": TargetSweeps("run_scripts.hyflexpim_greedy_sweep", "run_scripts.hyflexpim_listsched_sweep",
                              "run_scripts.hyflexpim_cpsat_sweep", "--models", []),
    # CP-SAT time units per ns: 1000 (ps), as the other targets' solve_capped default
    "rse": TargetSweeps("run_scripts.rse_greedy_sweep", "run_scripts.rse_listsched_sweep",
                        "run_scripts.rse_cpsat_sweep", "--apps", ["--time-scale", "1000"]),
}
if set(SWEEPS) != set(TARGETS):
    raise RuntimeError(f"sweeps defined for {sorted(SWEEPS)}, the figure has {TARGETS}")
if set(CHECKPOINTS) != set(MODELS):
    raise RuntimeError(f"checkpoints built for {sorted(CHECKPOINTS)}, the figure has {MODELS}")


def summary_path(target: str, method: str, model: str) -> Path:
    """The sweep's per-model results file: greedy, listsched or cpsat."""
    suffix = {"greedy": "offload_policy", "listsched": "listsched", "cpsat": "cpsat"}[method]
    return RESULTS_DIR / f"{target}_{suffix}_{model}.json"


def cpsat_mapping_path(target: str, model: str, cap: int) -> Path:
    """The CP-SAT mapping the figure reads its latency from."""
    return RESULTS_DIR / f"{target}_cpsat" / model / f"cap_{cap}.json"


def has_three_caps(target: str, method: str):
    """done-check: every model's results file of `method` has its three caps."""
    def check() -> bool:
        for model in MODELS:
            path = summary_path(target, method, model)
            if not path.exists() or len(json.loads(path.read_text())["points"]) != 3:
                return False
        return True
    return check


def figure_rows() -> dict:
    """{target: {model: [row per cap]}} of the figure's saved data."""
    return json.loads(FIGURE_DATA.read_text())["data"]["targets"]


def _step(target: str, method: str, workers: int) -> Step:
    """The target's sweep of `method` over every model. --force deletes the sweep's
    results files and its mappings (run_scripts/results/<target>_<method>/) first."""
    s = SWEEPS[target]
    if method == "greedy":
        cmd = [PY, "-m", s.greedy, "--workers", str(workers), s.models_flag, *MODELS]
    elif method == "listsched":
        cmd = [PY, "-m", s.listsched, s.models_flag, *MODELS]
    else:
        cmd = [PY, "-m", s.cpsat, s.models_flag, *MODELS, "--caps", "min", "median", "max",
               "--time-limit", str(CPSAT_TIME_LIMIT_S), "--cpsat-workers", str(workers), *s.cpsat_extra]
    mappings = sorted((RESULTS_DIR / f"{target}_{method}").glob("*/cap_*.json"))
    return Step(f"{target}: {method}", cmd, REPO, has_three_caps(target, method), {},
                [summary_path(target, method, m) for m in MODELS] + mappings)


def _compare_exact(target: str, method: str) -> list[dict]:
    """greedy / list scheduling: deterministic, so each latency must equal the figure's."""
    rows = []
    figure = figure_rows()[target]
    for model in MODELS:
        points = json.loads(summary_path(target, method, model).read_text())["points"]
        for fig in figure[model]:
            ours, paper = points[str(fig["cap"])]["latency"], fig[method]["latency_ns"]
            rows.append({"target": target, "model": model, "cap": fig["cap"], "method": method,
                         "latency_ns": ours, "figure_latency_ns": paper,
                         "status": "match" if ours == paper else "DIFFERS"})
    return rows


def _compare_cpsat(target: str) -> list[dict]:
    """CP-SAT: not deterministic (a time limit, parallel workers, the machine), so each
    case gives its relative difference to the figure; MISSING where the figure has a
    latency and this run found no mapping."""
    rows = []
    figure = figure_rows()[target]
    for model in MODELS:
        for fig in figure[model]:
            path = cpsat_mapping_path(target, model, fig["cap"])
            ours = json.loads(path.read_text())["latency_ns"] if path.exists() else None
            paper = fig["cpsat"]["latency_ns"] or None   # the figure stores "none" as None or 0
            if paper is None:
                status = "no figure value" if ours is None else "new (figure has none)"
            elif ours is None:
                status = "MISSING"
            else:
                status = f"{100 * (ours - paper) / paper:+.2f}%"
            rows.append({"target": target, "model": model, "cap": fig["cap"], "method": "cpsat",
                         "latency_ns": ours, "figure_latency_ns": paper, "status": status})
    return rows


def run_method(target: str, method: str, description: str) -> None:
    """A <target>_<method>.py script: runs the sweep (skipped if its results are complete;
    --force deletes them first), writes reproduce/results/mapping_<target>_<method>.csv
    and its run log, and exits 1 on a greedy / list-scheduling point that differs from
    the figure or a CP-SAT case the figure has and this run does not."""
    parser = argparse.ArgumentParser(description=description, formatter_class=argparse.RawDescriptionHelpFormatter)
    if method == "greedy":
        parser.add_argument("--workers", type=int, required=True, help="parallel processes, one model each")
    elif method == "cpsat":
        parser.add_argument("--cpsat-workers", type=int, required=True, help="CP-SAT search workers")
    parser.add_argument("--force", action="store_true", help="delete this sweep's results and rerun it")
    args = parser.parse_args()
    # list scheduling runs in one process: it has no worker count
    workers = args.workers if method == "greedy" else args.cpsat_workers if method == "cpsat" else 0
    log = run_steps(checkpoint_steps(MODELS) + [_step(target, method, workers)], args.force)
    rows = _compare_cpsat(target) if method == "cpsat" else _compare_exact(target, method)
    name = f"mapping_{target}_{method}"
    write_csv(RESULTS / f"{name}.csv", rows)
    write_csv(RESULTS / f"{name}_steps.csv", log)
    bad = [r for r in rows if r["status"] in ("DIFFERS", "MISSING")]
    for r in rows:
        if method == "cpsat" or r in bad:
            print(f"  {r['status']:>22s}  {r['model']:13s} cap {r['cap']:>10}: {r['latency_ns']} "
                  f"(figure {r['figure_latency_ns']})")
    print(f"wrote {RESULTS / (name + '.csv')}")
    if bad:
        print(f"{len(bad)} of {len(rows)} points differ from or are missing in the figure", file=sys.stderr)
        sys.exit(1)
    print(f"{len(rows)} points checked against the figure")
