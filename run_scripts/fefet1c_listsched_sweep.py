"""Maps each kernel given with --models to 1F1C with list scheduling
(compiler/xdsl_list_schedule_1f1c.py) under the shared offload policy, at the caps the
greedy sweep mapped it at (run_scripts/fefet1c_greedy_sweep.py's
results/fefet1c_offload_policy_<model>.json; raises if that file is missing), and writes
the latency, array counts and full mapping per cap.

Usage:
  python3 -m run_scripts.fefet1c_listsched_sweep --models mobilenet-v2 resnet50
Writes run_scripts/results/fefet1c_listsched_<model>.json and, per cap, the mapping
(compiler/xdsl_mapping_export.py) to run_scripts/results/fefet1c_listsched/<model>/cap_<cap>.json.
"""
import argparse
import json
import os
import time

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "1fefet1c/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_list_schedule_1f1c import list_schedule_1f1c
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import Budget
from kernels.registry import KNOWN_MODELS
from run_scripts.offload_policy_common import offload_policy_flags
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "fefet1c_listsched")


def greedy_caps(model: str) -> list[int]:
    """The caps the greedy sweep mapped `model` at."""
    path = os.path.join(RESULTS_DIR, f"fefet1c_offload_policy_{model}.json")
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path}: run run_scripts.fefet1c_greedy_sweep for {model} first")
    with open(path) as f:
        return sorted(int(c) for c in json.load(f)["points"])


def run_model(model: str) -> None:
    flags = offload_policy_flags()
    t0_wall = time.time()
    block = build_tiled_cim_block(model, flags)
    points = {}
    out_path = os.path.join(RESULTS_DIR, f"fefet1c_listsched_{model}.json")
    for cap in greedy_caps(model):
        t0 = time.time()
        result = list_schedule_1f1c(TARGET, block, Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={}))
        counts = result.array_counts
        mapping_path = os.path.join(MAPPINGS_DIR, model, f"cap_{cap}.json")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump({"target": TARGET.name, "model": model, "cap": cap, "flags": flags,
                       **export_mapping(block, result)}, f, allow_nan=False)
        points[str(cap)] = {"cap": cap, "latency": result.latency_ns, "arrays": counts.total_array_count,
                            "nonvolatile_arrays": counts.nonvolatile_array_count,
                            "volatile_arrays": counts.volatile_array_count,
                            "feasible": counts.total_array_count <= cap, "mapping": mapping_path}
        with open(out_path, "w") as f:
            json.dump({"target": TARGET.name, "model": model, "algorithm": "list_schedule_1f1c",
                       "policy": "matmul_matvec_i8_rest_digital",
                       "code_fingerprint": report_code_fingerprint(os.path.abspath(__file__)),
                       "points": points}, f, indent=2)
        print(f"[{model}] cap {cap}: {result.latency_ns / 1e6:.3f} ms, {counts.total_array_count} arrays, "
              f"{time.time() - t0:.1f}s", flush=True)
    print(f"[{model}] done: {len(points)} caps, wall={time.time() - t0_wall:.1f}s", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", required=True, choices=KNOWN_MODELS, help="kernels to map")
    args = parser.parse_args()
    for model in args.models:
        run_model(model)


if __name__ == "__main__":
    main()
