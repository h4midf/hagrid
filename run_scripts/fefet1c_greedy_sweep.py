"""Maps each kernel given with --models to 1F1C under the shared offload policy
(run_scripts/offload_policy_common.offload_policy_flags), at three array caps: the
smallest, the median and the largest of the cap sequence
(compiler.array_caps.build_cap_sequence), and writes the latency, array count and full
mapping per cap. The models are a command-line argument, not a list in this file: this
file is part of each result's code fingerprint, so choosing models must not change it.

Usage:
  python3 -m run_scripts.fefet1c_greedy_sweep --workers 4 --models mobilenet-v2 resnet50
Writes run_scripts/results/fefet1c_offload_policy_<model>.json (one
file per application, resumable per cap) and, per cap, the mapping
(compiler/xdsl_mapping_export.py) to
run_scripts/results/fefet1c_greedy/<model>/cap_<cap>.json.
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "1fefet1c/default"

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import Budget, _defining_op, initial_mapping, rebuild_hardware, run_capped_search_sync_group
from kernels.registry import KNOWN_MODELS
from run_scripts.offload_policy_common import load_if_complete, offload_policy_flags, select_min_median_max_caps
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# one mapping file per (kernel, cap): MAPPINGS_DIR/<model>/cap_<cap>.json
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "fefet1c_greedy")


def run_model_sweep(model: str, flags: dict, out_path: str) -> None:
    """Runs the capped search at every cap. Skips caps already in out_path,
    and the whole kernel if every cap is done."""
    code_fp = report_code_fingerprint(os.path.abspath(__file__))
    existing_data = load_if_complete(out_path, cap_selector=select_min_median_max_caps, code_fp=code_fp)
    if existing_data is not None:
        print(f"[{model}] already complete ({len(existing_data['points'])} caps) -- skipping", flush=True)
        return

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_data = json.load(f)
        if existing_data.get("code_fingerprint") != code_fp:
            print(f"[{model}] report code changed since {out_path} was last written -- "
                  f"discarding {len(existing_data.get('points', {}))} cached caps, recomputing fresh", flush=True)
        else:
            existing = existing_data.get("points", {})

    t0_wall = time.time()
    block = build_tiled_cim_block(model, flags)
    mapper, hardware = initial_mapping(TARGET, block)
    min_cap = int(rebuild_hardware(hardware).total_array_count)

    free_values = [
        v for v in mapper.node_order
        if not isinstance(v, BlockArgument) and isinstance(_defining_op(v), MAPPED_OPERATION_OPS)
    ]

    def _reset_to_pf1():
        for v in free_values:
            mapper.pf_overrides[mapper.node_index[v]] = 1
        mapper.compute_all()
        return rebuild_hardware(hardware)

    for v in free_values:
        mapper.pf_overrides[mapper.node_index[v]] = mapper.pf_cap_for_value(v)
    mapper.compute_all()
    max_cap = int(rebuild_hardware(hardware).total_array_count)
    _reset_to_pf1()

    caps = select_min_median_max_caps(build_cap_sequence(min_cap, max_cap))
    points = dict(existing)
    for cap in caps:
        if str(cap) in points:
            continue
        result = run_capped_search_sync_group(
            TARGET, block, Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={}),
            warm_start_pf={}, shared_caches={}, verbose=False,
        )
        arr = result.array_counts.total_array_count
        mapping_path = os.path.join(MAPPINGS_DIR, model, f"cap_{cap}.json")
        os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
        with open(mapping_path, "w") as f:
            json.dump({"target": TARGET.name, "model": model, "cap": cap, "flags": flags,
                       **export_mapping(block, result)}, f, allow_nan=False)
        points[str(cap)] = {"cap": cap, "latency": result.latency_ns, "arrays": arr, "feasible": arr <= cap,
                            "mapping": mapping_path}
        _reset_to_pf1()
        with open(out_path, "w") as f:
            json.dump({
                "target": "1fefet1c", "model": model, "policy": "matmul_matvec_i8_rest_digital",
                "code_fingerprint": code_fp,
                "min_feasible_array_cap": min_cap, "max_cap": max_cap,
                "points": points,
            }, f, indent=2)

    print(f"[{model}] done: {len(points)} caps ({min_cap}..{max_cap}), "
          f"wall={time.time() - t0_wall:.1f}s -> {out_path}", flush=True)


def _worker(model: str) -> str:
    flags = offload_policy_flags()
    out_path = os.path.join(RESULTS_DIR, f"fefet1c_offload_policy_{model}.json")
    run_model_sweep(model, flags, out_path)
    return model


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--models", nargs="+", required=True, choices=KNOWN_MODELS, help="kernels to map")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_worker, m): m for m in args.models}
        while pending:
            done, _not_done = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                model = pending.pop(fut)
                fut.result()
                print(f"  [{model}] complete", flush=True)


if __name__ == "__main__":
    main()
