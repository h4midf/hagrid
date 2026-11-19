"""Maps the bert-base variants (variant_generators/bert_base_quant.py) whose masked-LM
pseudo-perplexity is below --ppl-max (run_scripts/results/bert_base_variants.json,
bert_base_variants_merge.py) to one sized 1F1C config, each at its smallest array cap,
and gives each one's latency, energy and area. A variant at or above --ppl-max is never
compiled.

The config is TARGET_NAME (environment variable, required; targets.target_support reads
it at import time): 1fefet1c/64 or 1fefet1c/128 (targets/1fefet1c/<config>).

Per variant (its kernel and flags from the variants file):
  smallest cap  the smallest of the cap sequence (compiler.array_caps.build_cap_sequence
                between min_cap, every free op at pf = 1, and max_cap, every free op at
                its pf cap), as fefet1c_greedy_sweep.py
  mapping       the capped search at that cap (compiler/xdsl_mapping_export.py)
  placement     fefet1c_tiles.place_mapping
  energy        fefet1c_energy.energy_breakdown (one inference)
  area          fefet1c_area.area_breakdown

Writes, under run_scripts/results/fefet1c_bert_base_variants/<config>/:
  mappings/<variant>.json   the mapping
  <variant>.json            cap, latency, arrays, tiles, energy and area (totals and parts)
Resumable: a variant whose result file exists is skipped.

Usage (from the repo root):
  TARGET_NAME=1fefet1c/64 python3 -m run_scripts.fefet1c_bert_base_variant_sweep --ppl-max 14.4 --workers 6
"""
import argparse
import json
import multiprocessing
import os
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait

from xdsl.ir import BlockArgument

from compiler.array_caps import build_cap_sequence
from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import per_kind_accel_rows
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_search import Budget, _defining_op, initial_mapping, rebuild_hardware, run_capped_search_sync_group
from run_scripts.fefet1c_area import area_breakdown
from run_scripts.fefet1c_energy import energy_breakdown
from run_scripts.fefet1c_tiles import RESULTS_DIR, place_mapping
from run_scripts.offload_policy_common import select_min_median_max_caps
from targets.target_support import TARGET

CONFIGS = ("1fefet1c/64", "1fefet1c/128")
VARIANTS_FILE = os.path.join(RESULTS_DIR, "bert_base_variants.json")
OUT_ROOT = os.path.join(RESULTS_DIR, "fefet1c_bert_base_variants")


def out_dir() -> str:
    return os.path.join(OUT_ROOT, TARGET.name.split("/", 1)[1])


def smallest_cap(block) -> tuple[int, int, int]:
    """(min_cap, max_cap, the smallest cap of the cap sequence) of one block."""
    mapper, hardware = initial_mapping(TARGET, block)
    min_cap = int(rebuild_hardware(hardware).total_array_count)
    free = [v for v in mapper.node_order
            if not isinstance(v, BlockArgument) and isinstance(_defining_op(v), MAPPED_OPERATION_OPS)]
    for v in free:
        mapper.pf_overrides[mapper.node_index[v]] = mapper.pf_cap_for_value(v)
    mapper.compute_all()
    max_cap = int(rebuild_hardware(hardware).total_array_count)
    return min_cap, max_cap, select_min_median_max_caps(build_cap_sequence(min_cap, max_cap))[0]


def run_variant(name: str, model: str, flags: dict, ppl: float) -> str:
    t0 = time.time()
    block = build_tiled_cim_block(model, flags)
    min_cap, max_cap, cap = smallest_cap(block)
    result = run_capped_search_sync_group(
        TARGET, block, Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={}),
        warm_start_pf={}, shared_caches={}, verbose=False,
    )
    mapping_path = os.path.join(out_dir(), "mappings", f"{name}.json")
    with open(mapping_path, "w") as f:
        json.dump({"target": TARGET.name, "model": model, "cap": cap, "flags": flags,
                   **export_mapping(block, result)}, f, allow_nan=False)
    # placement, energy and area read the mapping as fefet1c_tiles / _energy / _area do
    with open(mapping_path) as f:
        mapping = json.load(f)
    tiles = {"target": TARGET.name, "model": model, "cap": cap, "mapping": mapping_path,
             "latency_ns": mapping["latency_ns"], **place_mapping(mapping)}
    unit_rows = per_kind_accel_rows(block)
    energy = energy_breakdown(mapping, tiles, unit_rows)
    area = area_breakdown(tiles, unit_rows)
    e_total, a_total = sum(energy.values()), sum(area.values())
    out = {
        "target": TARGET.name, "variant": name, "model": model, "flags": flags, "ppl": ppl,
        "min_cap": min_cap, "max_cap": max_cap, "cap": cap,
        "latency_ns": mapping["latency_ns"], "arrays": result.array_counts.total_array_count,
        "tiles": tiles["tiles"], "routers": tiles["routers"], "added_arrays": tiles["added_arrays"],
        "energy_mJ": e_total / 1e9,
        "energy_parts_mJ": {k: v / 1e9 for k, v in sorted(energy.items(), key=lambda kv: -kv[1])},
        "area_mm2": a_total / 1e6,
        "area_parts_mm2": {k: v / 1e6 for k, v in area.items()},
        "mapping": mapping_path,
    }
    with open(os.path.join(out_dir(), f"{name}.json"), "w") as f:
        json.dump(out, f, indent=2)
    return (f"{name}: cap {cap}, {mapping['latency_ns'] / 1e6:.3f} ms, {e_total / 1e9:.3f} mJ, "
            f"{a_total / 1e6:.2f} mm^2 ({time.time() - t0:.0f}s)")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--ppl-max", type=float, required=True, help="only variants with ppl below this run")
    parser.add_argument("--workers", type=int, required=True)
    args = parser.parse_args()
    if TARGET.name not in CONFIGS:
        raise ValueError(f"TARGET_NAME is {TARGET.name}; this sweep runs on {CONFIGS}")

    with open(VARIANTS_FILE) as f:
        variants = json.load(f)["variants"]
    todo = {n: e for n, e in variants.items() if e["ppl"] < args.ppl_max}
    os.makedirs(os.path.join(out_dir(), "mappings"), exist_ok=True)
    done = {n for n in todo if os.path.exists(os.path.join(out_dir(), f"{n}.json"))}
    print(f"{TARGET.name}: {len(todo)} of {len(variants)} variants below ppl {args.ppl_max}, "
          f"{len(done)} already done", flush=True)

    with ProcessPoolExecutor(max_workers=args.workers, mp_context=multiprocessing.get_context("spawn")) as pool:
        pending = {pool.submit(run_variant, n, e["model"], e["flags"], e["ppl"]): n
                   for n, e in todo.items() if n not in done}
        finished = len(done)
        while pending:
            ready, _ = wait(pending.keys(), return_when=FIRST_COMPLETED)
            for fut in ready:
                pending.pop(fut)
                finished += 1
                print(f"[{finished}/{len(todo)}] {fut.result()}", flush=True)


if __name__ == "__main__":
    main()
