"""Maps each app given with --apps to the RRAM/SRAM ensemble (targets/rram-sram-ensemble/
default, an EnsembleTarget bound to the app's SRAM correction share) under the shared
offload policy, at three macro caps: the smallest, the median and the largest of the
cap sequence (compiler.array_caps.build_cap_sequence), and writes the latency, macro
counts and full mapping per cap. The kernel is the app's group-wise 3-bit form
(<app>-w3, kernels/w3.py). A cap bounds the RRAM + SRAM macros (EnsembleBudget). Timing
only: area and energy are separate post-mapping scripts.

Usage:
  python3 -m run_scripts.rse_greedy_sweep --workers 1 --apps resnet50
Writes run_scripts/results/rse_offload_policy_<app>.json (one file per app, resumable
per cap) and, per cap, the mapping to run_scripts/results/rse_greedy/<app>/cap_<cap>.json.
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import (
    EnsembleBudget, _defining_op, initial_mapping, rebuild_hardware, run_capped_search_sync_group,
)
from kernels.registry import KNOWN_MODELS, W3_SUFFIX
from run_scripts.offload_policy_common import load_if_complete, offload_policy_flags, select_min_median_max_caps
from targets.target_loader import ensemble_for_app
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# one mapping file per (app, cap): MAPPINGS_DIR/<app>/cap_<cap>.json
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "rse_greedy")


def run_app_sweep(app: str, flags: dict, out_path: str) -> None:
    """Runs the capped search at every cap. Skips caps already in out_path,
    and the whole app if every cap is done."""
    code_fp = report_code_fingerprint(os.path.abspath(__file__))
    existing_data = load_if_complete(out_path, cap_selector=select_min_median_max_caps, code_fp=code_fp)
    if existing_data is not None:
        print(f"[{app}] already complete ({len(existing_data['points'])} caps) -- skipping", flush=True)
        return

    existing = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            existing_data = json.load(f)
        if existing_data.get("code_fingerprint") != code_fp:
            print(f"[{app}] report code changed since {out_path} was last written -- "
                  f"discarding {len(existing_data.get('points', {}))} cached caps, recomputing fresh", flush=True)
        else:
            existing = existing_data.get("points", {})

    t0_wall = time.time()
    kernel = app + W3_SUFFIX
    target = ensemble_for_app(TARGET, app)
    block = build_tiled_cim_block(kernel, flags)
    mapper, hardware = initial_mapping(target, block)
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
            target, block, EnsembleBudget(macro_cap=cap, accel_area_caps={}),
            warm_start_pf={}, shared_caches={}, verbose=False,
        )
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
        _reset_to_pf1()
        with open(out_path, "w") as f:
            json.dump({
                "target": TARGET.name, "app": app, "model": kernel, "policy": "matmul_matvec_i8_rest_digital",
                "sram_share_pct": target.sram_share_pct, "code_fingerprint": code_fp,
                "min_feasible_array_cap": min_cap, "max_cap": max_cap,
                "points": points,
            }, f, indent=2)
        print(f"[{app}] cap {cap}: {result.latency_ns / 1e6:.3f} ms, {counts.total_array_count} macros "
              f"(RRAM {counts.rram.total_array_count}, SRAM {counts.sram.total_array_count})", flush=True)

    print(f"[{app}] done: {len(points)} caps ({min_cap}..{max_cap}), "
          f"wall={time.time() - t0_wall:.1f}s -> {out_path}", flush=True)


def _worker(app: str) -> str:
    flags = offload_policy_flags()
    out_path = os.path.join(RESULTS_DIR, f"rse_offload_policy_{app}.json")
    run_app_sweep(app, flags, out_path)
    return app


def main() -> None:
    apps = sorted(TARGET.sram_correction_weight_pct)
    for app in apps:
        if app + W3_SUFFIX not in KNOWN_MODELS:
            raise KeyError(f"app {app!r} has no {W3_SUFFIX} kernel")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--workers", type=int, required=True)
    parser.add_argument("--apps", nargs="+", required=True, choices=apps, help="apps to map")
    args = parser.parse_args()

    os.makedirs(RESULTS_DIR, exist_ok=True)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(_worker, a): a for a in args.apps}
        while pending:
            done, _not_done = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in done:
                app = pending.pop(fut)
                fut.result()
                print(f"  [{app}] complete", flush=True)


if __name__ == "__main__":
    main()
