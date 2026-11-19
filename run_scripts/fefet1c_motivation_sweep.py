"""The paper's mapping motivation (background, Fig. 3) on 1F1C: the 12-layer attention app
(bert-attention-12layer, kernels/bert_attention_only.py) under the shared offload policy,
mapped at five array caps of its range (compiler.array_caps.select_five_caps: the smallest
and largest power of two, three between at near-equal steps; --caps picks others) by three
methods:

  greedy     run_capped_search_sync_group (CP-SAT's per-segment stationary caps come from it)
  listsched  compiler/xdsl_list_schedule_1f1c.py
  cpsat      compiler/xdsl_cpsat.solve_capped, one 50-op layer solved and copied to all 12

Writes run_scripts/results/fefet1c_motivation_bert-attention-12layer.json (resumable per
cap) and, per method and cap, the mapping (compiler/xdsl_mapping_export.py) to
run_scripts/results/fefet1c_motivation/<method>/cap_<cap>.json.

Usage:
  python3 -m run_scripts.fefet1c_motivation_sweep --time-limit 1800 --cpsat-workers 8
  python3 -m run_scripts.fefet1c_motivation_sweep --caps 32768 524288 --time-limit 1800 --cpsat-workers 8
"""
import argparse
import json
import os
import time

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "1fefet1c/default"

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence, select_five_caps
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_cpsat import solve_capped
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_list_schedule_1f1c import list_schedule_1f1c
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import Budget, _defining_op, initial_mapping, rebuild_hardware, run_capped_search_sync_group
from compiler.xdsl_segments import find_segments, stationary_by_segment
from run_scripts.offload_policy_common import offload_policy_flags
from targets.target_support import TARGET

APP = "bert-attention-12layer"
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS_DIR, f"fefet1c_motivation_{APP}.json")
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "fefet1c_motivation")


def _cap_range(block) -> tuple[int, int]:
    """Total arrays with every array op at pf 1 and at its pf cap, as fefet1c_greedy_sweep."""
    mapper, hardware = initial_mapping(TARGET, block)
    lo = int(rebuild_hardware(hardware).total_array_count)
    for v in mapper.node_order:
        if not isinstance(v, BlockArgument) and isinstance(_defining_op(v), MAPPED_OPERATION_OPS):
            mapper.pf_overrides[mapper.node_index[v]] = mapper.pf_cap_for_value(v)
    mapper.compute_all()
    return lo, int(rebuild_hardware(hardware).total_array_count)


def _save_mapping(block, flags: dict, method: str, cap: int, result) -> str:
    path = os.path.join(MAPPINGS_DIR, method, f"cap_{cap}.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({"target": TARGET.name, "model": APP, "method": method, "cap": cap, "flags": flags,
                   **export_mapping(block, result)}, f, allow_nan=False)
    return os.path.relpath(path, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo-relative


def _entry(result, cap: int, wall_s: float, mapping: str) -> dict:
    counts = result.array_counts
    return {"latency": result.latency_ns, "arrays": counts.total_array_count,
            "stationary_arrays": counts.nonvolatile_array_count, "wmu_peak_arrays": counts.volatile_array_count,
            "feasible": counts.total_array_count <= cap, "wall_s": wall_s, "mapping": mapping}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--caps", nargs="+", type=int, default=None, help="array caps (default: the app's range)")
    parser.add_argument("--time-limit", type=float, required=True, help="CP-SAT seconds per segment")
    parser.add_argument("--cpsat-workers", type=int, required=True, help="CP-SAT search workers")
    parser.add_argument("--time-scale", type=int, default=1000, help="CP-SAT time units per ns (1000 = ps)")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)

    flags = offload_policy_flags()
    code_fp = report_code_fingerprint(os.path.abspath(__file__))
    points = {}
    if os.path.exists(OUT_PATH):
        with open(OUT_PATH) as f:
            data = json.load(f)
        if data.get("code_fingerprint") == code_fp:
            points = data["points"]
        else:
            print(f"code changed since {OUT_PATH} was written -- recomputing every cap", flush=True)

    block = build_tiled_cim_block(APP, flags)
    segments = find_segments(TARGET, block, initial_mapping(TARGET, block)[0])
    min_cap, max_cap = _cap_range(block)
    caps = args.caps or select_five_caps(build_cap_sequence(min_cap, max_cap))
    print(f"[{APP}] segments {[(len(s.region_values), s.repeats) for s in segments]}, "
          f"range {min_cap}..{max_cap}, caps {caps}", flush=True)

    for cap in caps:
        if str(cap) in points:
            print(f"[{APP}] cap {cap} already done -- skipping", flush=True)
            continue
        point = {"cap": cap}
        budget = Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={})

        t0 = time.time()
        greedy = run_capped_search_sync_group(TARGET, block, budget, warm_start_pf={}, shared_caches={},
                                              verbose=False)
        point["greedy"] = _entry(greedy, cap, time.time() - t0, _save_mapping(block, flags, "greedy", cap, greedy))

        t0 = time.time()
        try:
            ls = list_schedule_1f1c(TARGET, block, budget)
        except RuntimeError as e:
            point["listsched"] = {"status": "failed", "error": str(e), "wall_s": time.time() - t0}
        else:
            point["listsched"] = _entry(ls, cap, time.time() - t0, _save_mapping(block, flags, "listsched", cap, ls))

        split = stationary_by_segment(segments, greedy.hardware)
        t0 = time.time()
        try:
            solved = solve_capped(TARGET, block, cap, segments, split, time_limit_s=args.time_limit,
                                  workers=args.cpsat_workers, scale=args.time_scale)
        except RuntimeError as e:
            # no CP-SAT mapping: greedy's mapping stands in for it (marked "source": "greedy")
            point["cpsat"] = {**_entry(greedy, cap, time.time() - t0, _save_mapping(block, flags, "cpsat", cap, greedy)),
                              "source": "greedy", "cpsat_error": str(e), "time_scale": args.time_scale}
        else:
            r = solved.result
            point["cpsat"] = {**_entry(r, cap, time.time() - t0, _save_mapping(block, flags, "cpsat", cap, r)),
                              "source": "cpsat", "status": solved.status, "time_scale": args.time_scale,
                              "segment_stationary_caps": [s.cap_per_copy for s in split],
                              "segments": [{"status": s.status, "objective_ns": s.objective_ns,
                                            "bound_ns": s.bound_ns, "solve_wall_s": s.solve_wall_s}
                                           for s in solved.segments]}

        points[str(cap)] = point
        with open(OUT_PATH, "w") as f:
            json.dump({"target": TARGET.name, "model": APP, "policy": "matmul_matvec_i8_rest_digital",
                       "code_fingerprint": code_fp, "min_feasible_array_cap": min_cap, "max_cap": max_cap,
                       "points": points}, f, indent=2)
        summary = ", ".join(f"{m} {point[m]['latency']:.0f} ns" if "latency" in point[m] else f"{m} failed"
                            for m in ("greedy", "listsched", "cpsat"))
        print(f"[{APP}] cap {cap}: {summary}", flush=True)


if __name__ == "__main__":
    main()
