"""Dumps the IR of the active kernel at each dialect boundary:

  01_gen.mlir    -- the traced model in the gen dialect.
  02_cinm.mlir   -- after the gen -> cinm lowering and cinm canonicalization.
  03_cim.mlir    -- after the cinm -> cim lowering, stationary conversion and tiling.
  04_mapped.mlir -- cim IR with the searched mapping as mapped.* ops.
  05_hcim.mlir   -- the searched mapping as hcim.* device instructions.

One trace, one lowering and one search produce all five files.

Inputs: a variant file written by variant_generators/<model>.py, whose "config"
({fx_name: per-op config}) drives every pass, and the array cap the search maps
under (04 and 05).

Usage (from the repo root):
  ACTIVE_MODEL=bert-large TARGET_NAME=1fefet1c/default python3 -m compiler.dump_ir \\
      dump/bert_large_variant_files_sharded/00/int8_wchan_achan_asym_full_dw16__sm-exact_ge-exact__rw512-tree_mw1024_aw1024.json \\
      --array-cap 4096
"""

import argparse
import json
import os

from compiler.lowering import func_block, lower_to_cinm, lower_to_tiled_cim, trace_kernel
from compiler.xdsl_bridge import gen_fx_to_xdsl
from compiler.xdsl_hcim_materialize import materialize_hcim
from compiler.xdsl_materialize import materialize_schedule
from compiler.xdsl_search import Budget, run_capped_search_sync_group
from kernels.registry import KNOWN_MODELS
from targets.target_support import TARGET

# the model to dump (environment variable, required)
ACTIVE_MODEL = os.environ["ACTIVE_MODEL"]
if ACTIVE_MODEL not in KNOWN_MODELS:
    raise ValueError(f"ACTIVE_MODEL {ACTIVE_MODEL!r} is not one of {KNOWN_MODELS}")

DUMP_ROOT = "dump"


def _write(module, path: str, header: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(f"// {header}\n")
        f.write(str(module))
        f.write("\n")
    print(f"wrote {path}")


def _load_variant_config(path: str) -> dict:
    """A variant file holds one entry: {variant_name: {"config": ..., "result": ...}}."""
    with open(path) as f:
        (entry,) = json.load(f).values()
    return entry["config"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("variant_file", help="variant JSON written by variant_generators/<model>.py")
    parser.add_argument("--array-cap", type=int, required=True, help="array cap the search maps under")
    args = parser.parse_args()

    config = _load_variant_config(args.variant_file)
    out_dir = os.path.join(DUMP_ROOT, ACTIVE_MODEL)

    traced = trace_kernel(ACTIVE_MODEL)
    _write(gen_fx_to_xdsl(traced), os.path.join(out_dir, "01_gen.mlir"), f"{ACTIVE_MODEL}: gen dialect")

    module = lower_to_cinm(traced, config)
    _write(module, os.path.join(out_dir, "02_cinm.mlir"), f"{ACTIVE_MODEL}: cinm dialect")

    lower_to_tiled_cim(module, config, TARGET)
    _write(module, os.path.join(out_dir, "03_cim.mlir"), f"{ACTIVE_MODEL}: cim dialect, tiled")

    block = func_block(module)
    result = run_capped_search_sync_group(
        TARGET, block, Budget(nonvolatile_cap=args.array_cap, volatile_cap=args.array_cap, accel_area_caps={}),
        warm_start_pf={}, shared_caches={}, verbose=True,
    )
    mapper, hardware = result.mapper, result.hardware

    # materialize_hcim leaves the block untouched; materialize_schedule mutates it, so it runs last.
    hcim_module = materialize_hcim(block, mapper, hardware)
    _write(hcim_module, os.path.join(out_dir, "05_hcim.mlir"), f"{ACTIVE_MODEL}: hcim dialect")

    materialize_schedule(block, mapper, hardware)
    _write(module, os.path.join(out_dir, "04_mapped.mlir"), f"{ACTIVE_MODEL}: cim dialect + mapped schedule")


if __name__ == "__main__":
    main()
