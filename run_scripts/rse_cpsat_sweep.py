"""Maps each app given with --apps to the RRAM/SRAM ensemble (targets/rram-sram-ensemble/
default, an EnsembleTarget bound to the app's SRAM correction share) with CP-SAT
(compiler.xdsl_cpsat.solve_capped), under the shared offload policy and at the same macro
caps as rse_greedy_sweep.py (the smallest, the median and the largest of the cap
sequence; --caps picks a subset). The kernel is the app's group-wise 3-bit form (<app>-w3,
kernels/w3.py). A cap bounds the RRAM + SRAM macros (EnsembleBudget). CP-SAT searches the
pf of every matvec / lut_search from scratch, one distinct segment at a time (a repeated
layer is solved once and copied); every other op is treated as greedy treats it.
--time-limit is per cap (one case), split equally over its distinct segments. The same apps as the greedy and LS sweeps (CPSAT_APPS);
without --apps, all of them in turn.

Per cap, greedy runs first (run_capped_search_sync_group): its RRAM stationary macros and
its SRAM correction macros, split by segment (compiler.xdsl_segments.stationary_by_segment,
sram_correction_by_segment), are CP-SAT's fixed per-segment caps. Greedy's latency is
written next to CP-SAT's.

Usage:
  python3 -m run_scripts.rse_cpsat_sweep --time-limit 3600 --cpsat-workers 8
  python3 -m run_scripts.rse_cpsat_sweep --apps bert-base --time-limit 3600 --cpsat-workers 8
Writes run_scripts/results/rse_cpsat_<app>.json (resumable per cap) and, per cap, the
mapping (compiler/xdsl_mapping_export.py) to run_scripts/results/rse_cpsat/<app>/cap_<cap>.json.
"""
import argparse
import json
import os
import time

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_cpsat import solve_capped
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_neurosim_export import report_code_fingerprint
from compiler.xdsl_search import (
    EnsembleBudget, _defining_op, initial_mapping, rebuild_hardware, run_capped_search_sync_group,
)
from compiler.xdsl_segments import find_segments, sram_correction_by_segment, stationary_by_segment
from kernels.registry import KNOWN_MODELS, W3_SUFFIX
from run_scripts.offload_policy_common import offload_policy_flags, select_min_median_max_caps
from targets.target_loader import ensemble_for_app
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
# one mapping file per (app, cap): MAPPINGS_DIR/<app>/cap_<cap>.json
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "rse_cpsat")
CAP_NAMES = ("min", "median", "max")
# the apps this sweep maps: the same apps as the greedy and LS sweeps
CPSAT_APPS = ("bert-base", "vit-base", "gpt2", "bert-large", "vit-large", "vgg16", "resnet50", "mobilenet-v2")


def _cap_range(target, block) -> tuple[int, int]:
    """Total macros with every array op (matvec, lut_search) at pf 1 and at its pf cap,
    exactly as rse_greedy_sweep computes them."""
    mapper, hardware = initial_mapping(target, block)
    lo = int(rebuild_hardware(hardware).total_array_count)
    free = [v for v in mapper.node_order
            if not isinstance(v, BlockArgument) and isinstance(_defining_op(v), MAPPED_OPERATION_OPS)]
    for v in free:
        mapper.pf_overrides[mapper.node_index[v]] = mapper.pf_cap_for_value(v)
    mapper.compute_all()
    hi = int(rebuild_hardware(hardware).total_array_count)
    return lo, hi


def run_app(app: str, cap_names: list[str], time_limit_s: float, workers: int, log: bool, time_scale: int) -> None:
    flags = offload_policy_flags()
    kernel = app + W3_SUFFIX
    out_path = os.path.join(RESULTS_DIR, f"rse_cpsat_{app}.json")
    code_fp = report_code_fingerprint(os.path.abspath(__file__))
    points = {}
    if os.path.exists(out_path):
        with open(out_path) as f:
            data = json.load(f)
        if data.get("code_fingerprint") == code_fp:
            points = data.get("points", {})
        else:
            print(f"[{app}] code changed since {out_path} was written -- recomputing every cap", flush=True)

    target = ensemble_for_app(TARGET, app)
    block = build_tiled_cim_block(kernel, flags)
    segments = find_segments(target, block, initial_mapping(target, block)[0])
    segment_limit_s = time_limit_s / len(segments)
    print(f"[{app}] segments (ops x repeats): {[(len(s.region_values), s.repeats) for s in segments]}, "
          f"{segment_limit_s:.0f} s each", flush=True)
    min_cap, max_cap = _cap_range(target, block)
    selected = select_min_median_max_caps(build_cap_sequence(min_cap, max_cap))
    caps = [selected[CAP_NAMES.index(name)] for name in cap_names] if len(selected) == 3 else selected

    for cap in caps:
        if str(cap) in points:
            print(f"[{app}] cap {cap} already done -- skipping", flush=True)
            continue
        t0 = time.time()
        greedy = run_capped_search_sync_group(
            target, block, EnsembleBudget(macro_cap=cap, accel_area_caps={}),
            warm_start_pf={}, shared_caches={}, verbose=False,
        )
        greedy_wall = time.time() - t0
        rram_split = stationary_by_segment(segments, greedy.hardware)
        correction_split = sram_correction_by_segment(segments, greedy.hardware)
        t0 = time.time()
        point = {"cap": cap, "time_limit_s": time_limit_s, "segment_time_limit_s": segment_limit_s,
                 "workers": workers, "time_scale": time_scale,
                 "greedy_latency": greedy.latency_ns, "greedy_arrays": greedy.array_counts.total_array_count,
                 "greedy_wall_s": greedy_wall,
                 "segment_rram_stationary_caps": [s.cap_per_copy for s in rram_split],
                 "segment_sram_correction_caps": [c.cap_per_copy for c in correction_split]}
        try:
            solved = solve_capped(target, block, cap, segments, rram_split, time_limit_s=segment_limit_s,
                                  workers=workers, scale=time_scale, verbose=log, correction_caps=correction_split)
        except RuntimeError as e:
            point.update({"status": "failed", "error": str(e), "wall_s": time.time() - t0})
            print(f"[{app}] cap {cap}: CP-SAT failed: {e}", flush=True)
        else:
            r = solved.result
            counts = r.array_counts
            mapping_path = os.path.join(MAPPINGS_DIR, app, f"cap_{cap}.json")
            os.makedirs(os.path.dirname(mapping_path), exist_ok=True)
            with open(mapping_path, "w") as f:
                json.dump({"target": TARGET.name, "app": app, "model": kernel, "cap": cap,
                           "sram_share_pct": target.sram_share_pct, "flags": flags,
                           **export_mapping(block, r)}, f, allow_nan=False)
            point.update({
                "status": solved.status, "latency": r.latency_ns, "arrays": counts.total_array_count,
                "rram_macros": counts.rram.total_array_count,
                "rram_stationary_macros": counts.rram.nonvolatile_array_count,
                "sram_macros": counts.sram.total_array_count,
                "sram_correction_macros": counts.sram.nonvolatile_array_count,
                "feasible": counts.total_array_count <= cap,
                "segments": [{"status": s.status, "objective_ns": s.objective_ns, "bound_ns": s.bound_ns,
                              "solve_wall_s": s.solve_wall_s} for s in solved.segments],
                "solve_wall_s": solved.solve_wall_s, "wall_s": time.time() - t0, "mapping": mapping_path,
            })
            print(f"[{app}] cap {cap}: CP-SAT {solved.status} {r.latency_ns:.0f} ns vs greedy "
                  f"{greedy.latency_ns:.0f} ns, solve {solved.solve_wall_s:.0f}s", flush=True)
        points[str(cap)] = point
        with open(out_path, "w") as f:
            json.dump({"target": TARGET.name, "app": app, "model": kernel, "policy": "matmul_matvec_i8_rest_digital",
                       "sram_share_pct": target.sram_share_pct, "code_fingerprint": code_fp,
                       "min_feasible_array_cap": min_cap, "max_cap": max_cap, "points": points}, f, indent=2)


def main() -> None:
    for app in CPSAT_APPS:
        if app + W3_SUFFIX not in KNOWN_MODELS:
            raise KeyError(f"app {app!r} has no {W3_SUFFIX} kernel")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--apps", nargs="+", choices=CPSAT_APPS, default=list(CPSAT_APPS),
                        help="apps to map, in order (default: all of them)")
    parser.add_argument("--caps", nargs="+", choices=CAP_NAMES, default=list(CAP_NAMES),
                        help="which of the three caps to run (default: all)")
    parser.add_argument("--time-limit", type=float, required=True,
                        help="CP-SAT seconds per cap, split over its distinct segments")
    parser.add_argument("--cpsat-workers", type=int, required=True, help="CP-SAT search workers")
    parser.add_argument("--log", action="store_true", help="print CP-SAT's search log")
    parser.add_argument("--time-scale", type=int, default=1000,
                        help="CP-SAT time units per ns (1000 = ps, 1 = ns; default 1000)")
    args = parser.parse_args()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    # one app at a time: CP-SAT already uses --cpsat-workers cores
    for app in args.apps:
        run_app(app, args.caps, args.time_limit, args.cpsat_workers, args.log, args.time_scale)


if __name__ == "__main__":
    main()
