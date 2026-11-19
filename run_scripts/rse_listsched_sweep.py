"""Maps each app given with --apps to the RRAM/SRAM ensemble with list scheduling
(compiler/xdsl_list_schedule_rse.py), the target bound to the app's SRAM correction share,
under the shared offload policy, at the caps the greedy sweep mapped it at
(run_scripts/rse_greedy_sweep.py's results/rse_offload_policy_<app>.json; raises if that
file is missing), and writes the latency, macro counts and full mapping per cap. The kernel
is the app's group-wise 3-bit form (<app>-w3, kernels/w3.py). A cap bounds the RRAM + SRAM
macros (EnsembleBudget).

Usage:
  python3 -m run_scripts.rse_listsched_sweep --apps resnet50
Writes run_scripts/results/rse_listsched_<app>.json and, per cap, the mapping
(compiler/xdsl_mapping_export.py) to run_scripts/results/rse_listsched/<app>/cap_<cap>.json.
"""
import argparse
import json
import os
import time

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_list_schedule_rse import list_schedule_rse
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import EnsembleBudget
from kernels.registry import W3_SUFFIX
from run_scripts.offload_policy_common import offload_policy_flags
from targets.target_loader import ensemble_for_app
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "rse_listsched")


def greedy_caps(app: str) -> list[int]:
    """The caps the greedy sweep mapped `app` at."""
    path = os.path.join(RESULTS_DIR, f"rse_offload_policy_{app}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run run_scripts.rse_greedy_sweep for {app} first")
    with open(path) as f:
        return sorted(int(c) for c in json.load(f)["points"])


def run_app(app: str) -> None:
    flags = offload_policy_flags()
    kernel = app + W3_SUFFIX
    target = ensemble_for_app(TARGET, app)
    t0_wall = time.time()
    block = build_tiled_cim_block(kernel, flags)
    points = {}
    out_path = os.path.join(RESULTS_DIR, f"rse_listsched_{app}.json")
    for cap in greedy_caps(app):
        t0 = time.time()
        result = list_schedule_rse(target, block, EnsembleBudget(macro_cap=cap, accel_area_caps={}))
        counts = result.array_counts
        mapping_path = os.path.join(MAPPINGS_DIR, app, f"cap_{cap}.json")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump({"target": TARGET.name, "app": app, "model": kernel, "cap": cap,
                       "sram_share_pct": target.sram_share_pct, "flags": flags,
                       **export_mapping(block, result)}, f, allow_nan=False)
        points[str(cap)] = {"cap": cap, "latency": result.latency_ns, "arrays": counts.total_array_count,
                            "rram_macros": counts.rram.total_array_count,
                            "sram_macros": counts.sram.total_array_count,
                            "feasible": counts.total_array_count <= cap, "mapping": mapping_path}
        with open(out_path, "w") as f:
            json.dump({"target": TARGET.name, "app": app, "model": kernel, "algorithm": "list_schedule_rse",
                       "policy": "matmul_matvec_i8_rest_digital", "sram_share_pct": target.sram_share_pct,
                       "code_fingerprint": report_code_fingerprint(os.path.abspath(__file__)),
                       "points": points}, f, indent=2)
        print(f"[{app}] cap {cap}: {result.latency_ns / 1e6:.3f} ms, {counts.total_array_count} macros, "
              f"{time.time() - t0:.1f}s", flush=True)
    print(f"[{app}] done: {len(points)} caps, wall={time.time() - t0_wall:.1f}s", flush=True)


def main() -> None:
    apps = sorted(TARGET.sram_correction_weight_pct)
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", nargs="+", required=True, choices=apps, help="apps to map")
    args = parser.parse_args()
    for app in args.apps:
        run_app(app)


if __name__ == "__main__":
    main()
