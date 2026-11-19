"""Maps each kernel given with --models to 1F1C with CP-SAT (compiler.xdsl_cpsat.
solve_capped), under the same offload policy and at the same array caps as
fefet1c_greedy_sweep.py (the smallest, the median and the largest of the cap sequence;
--caps picks a subset). CP-SAT searches the pf of every matvec / lut_search from
scratch, one distinct segment at a time (a repeated layer is solved once and copied);
every other op is treated as greedy treats it. --time-limit is per cap (one case), split
equally over its distinct segments.

Per cap, greedy runs first (run_capped_search_sync_group): its stationary arrays, split
by segment (compiler.xdsl_segments.stationary_by_segment), are CP-SAT's fixed
per-segment stationary caps. Greedy's latency is written next to CP-SAT's.

The same apps as the greedy and LS sweeps (CPSAT_MODELS); without --models, all of them in turn.

Usage:
  python3 -m run_scripts.fefet1c_cpsat_sweep --time-limit 3600 --cpsat-workers 8
  python3 -m run_scripts.fefet1c_cpsat_sweep --models bert-base --time-limit 3600 --cpsat-workers 8
  python3 -m run_scripts.fefet1c_cpsat_sweep --models bert-base --caps min --time-limit 3600 --cpsat-workers 8 --log
Writes run_scripts/results/fefet1c_cpsat_<model>.json (resumable per cap) and, per cap,
the mapping (compiler/xdsl_mapping_export.py) to
run_scripts/results/fefet1c_cpsat/<model>/cap_<cap>.json.
"""
import argparse
import json
import os
import time

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "1fefet1c/default"

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_cpsat import solve_capped
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import Budget, initial_mapping, rebuild_hardware, run_capped_search_sync_group
from compiler.xdsl_segments import find_segments, stationary_by_segment
from compiler.xdsl_timing import TimingOnlyMapping
from run_scripts.offload_policy_common import offload_policy_flags, select_min_median_max_caps
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# one mapping file per (kernel, cap): MAPPINGS_DIR/<model>/cap_<cap>.json
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "fefet1c_cpsat")
CAP_NAMES = ("min", "median", "max")
# the kernels this sweep maps: the same apps as the greedy and LS sweeps
CPSAT_MODELS = ("bert-base", "vit-base", "gpt2", "bert-large", "vit-large", "vgg16", "resnet50", "mobilenet-v2")


def _cap_range(block) -> tuple[int, int]:
    """Total arrays with every op at pf 1 and at its pf cap, as fefet1c_greedy_sweep."""
    mapper, hardware = initial_mapping(TARGET, block)
    ops = [i for i, v in enumerate(mapper.node_order) if not isinstance(v, BlockArgument)]
    mapper.pf_overrides = {i: 1 for i in ops}
    mapper.compute_all()
    lo = int(rebuild_hardware(hardware).total_array_count)
    mapper.pf_overrides = {i: mapper.pf_cap_at(i) for i in ops}
    mapper.compute_all()
    hi = int(rebuild_hardware(hardware).total_array_count)
    return lo, hi


def run_model(model: str, cap_names: list[str], time_limit_s: float, workers: int, log: bool) -> None:
    flags = offload_policy_flags()
    out_path = os.path.join(RESULTS_DIR, f"fefet1c_cpsat_{model}.json")
    code_fp = report_code_fingerprint(os.path.abspath(__file__))
    points = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        if data.get("code_fingerprint") == code_fp:
            points = data.get("points", {})
        else:
            print(f"[{model}] code changed since {out_path} was written -- recomputing every cap", flush=True)

    block = build_tiled_cim_block(model, flags)
    segments = find_segments(TARGET, block, TimingOnlyMapping(TARGET, block))
    segment_limit_s = time_limit_s / len(segments)
    print(f"[{model}] segments (ops x repeats): {[(len(s.region_values), s.repeats) for s in segments]}, "
          f"{segment_limit_s:.0f} s each", flush=True)
    min_cap, max_cap = _cap_range(block)
    selected = select_min_median_max_caps(build_cap_sequence(min_cap, max_cap))
    caps = [selected[CAP_NAMES.index(name)] for name in cap_names] if len(selected) == 3 else selected

    for cap in caps:
        if str(cap) in points:
            print(f"[{model}] cap {cap} already done -- skipping", flush=True)
            continue
        t0 = time.time()
        greedy = run_capped_search_sync_group(
            TARGET, block, Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={}),
            warm_start_pf={}, shared_caches={}, verbose=False,
        )
        greedy_wall = time.time() - t0
        split = stationary_by_segment(segments, greedy.hardware)
        t0 = time.time()
        point = {"cap": cap, "time_limit_s": time_limit_s, "segment_time_limit_s": segment_limit_s,
                 "workers": workers,
                 "greedy_latency": greedy.latency_ns, "greedy_arrays": greedy.array_counts.total_array_count,
                 "greedy_wall_s": greedy_wall,
                 "segment_stationary_caps": [s.cap_per_copy for s in split]}
        try:
            solved = solve_capped(TARGET, block, cap, segments, split, time_limit_s=segment_limit_s,
                                  workers=workers, verbose=log)
        except RuntimeError as e:
            point.update({"status": "failed", "error": str(e), "wall_s": time.time() - t0})
            print(f"[{model}] cap {cap}: CP-SAT failed: {e}", flush=True)
        else:
            r = solved.result
            mapping_path = os.path.join(MAPPINGS_DIR, model, f"cap_{cap}.json")
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, "w") as f:
                json.dump({"target": TARGET.name, "model": model, "cap": cap, "flags": flags,
                           **export_mapping(block, r)}, f, allow_nan=False)
            point.update({
                "status": solved.status, "latency": r.latency_ns, "arrays": r.array_counts.total_array_count,
                "stationary_arrays": r.array_counts.nonvolatile_array_count,
                "wmu_peak_arrays": r.array_counts.volatile_array_count,
                "feasible": r.array_counts.total_array_count <= cap,
                "segments": [{"status": s.status, "objective_ns": s.objective_ns, "bound_ns": s.bound_ns,
                              "solve_wall_s": s.solve_wall_s} for s in solved.segments],
                "solve_wall_s": solved.solve_wall_s, "wall_s": time.time() - t0, "mapping": mapping_path,
            })
            print(f"[{model}] cap {cap}: CP-SAT {solved.status} {r.latency_ns:.0f} ns vs greedy "
                  f"{greedy.latency_ns:.0f} ns, solve {solved.solve_wall_s:.0f}s", flush=True)
        points[str(cap)] = point
        with open(out_path, "w") as f:
            json.dump({"target": "1fefet1c", "model": model, "policy": "matmul_matvec_i8_rest_digital",
                       "code_fingerprint": code_fp, "min_feasible_array_cap": min_cap, "max_cap": max_cap,
                       "points": points}, f, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--models", nargs="+", choices=CPSAT_MODELS, default=list(CPSAT_MODELS),
                        help="kernels to map, in order (default: all of them)")
    parser.add_argument("--caps", nargs="+", choices=CAP_NAMES, default=list(CAP_NAMES),
                        help="which of the three caps to run (default: all)")
    parser.add_argument("--time-limit", type=float, required=True, help="CP-SAT seconds per cap, split over its distinct segments")
    parser.add_argument("--cpsat-workers", type=int, required=True, help="CP-SAT search workers")
    parser.add_argument("--log", action="store_true", help="print CP-SAT's search log")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # one kernel at a time: CP-SAT already uses --cpsat-workers cores
    for model in args.models:
        run_model(model, args.caps, args.time_limit, args.cpsat_workers, args.log)


if __name__ == "__main__":
    main()
