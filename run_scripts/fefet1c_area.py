"""Chip area of a placed 1F1C mapping, per part.

Runs on the tile files fefet1c_tiles.py writes (results/fefet1c_tiles/<model>/
cap_<cap>.json); it never maps, searches or places. Areas are at the target's node
(TARGET.node_nm):

  arrays              arrays x TARGET.array_area (cells, column ADCs, column front
                      ends, input / phase control, shift-and-add, WMU readout)
  added arrays        arrays the placement added beyond the search's count
                      (fefet1c_tiles.py step 2), x the same per-array area
  PE input registers  PEs x TARGET.organization.pe_input_register_area_um2
  tile buses          tiles x TARGET.organization.tile_bus_area_um2
  routers             routers x TARGET.organization.router_area_um2
  accelerator units   per-tile unit counts x each unit's area (with its lane
                      register), from xdsl_accel_concurrency.per_kind_accel_rows of the
                      model's block, rebuilt with the mapping's own flags

Not counted (targets/1fefet1c/default/organization.json "excluded"): off-chip I/O and
chip-level control. Writes results/fefet1c_area/<model>/cap_<cap>.json.

The target is TARGET_NAME (environment variable, required; targets.target_support reads
it at import time), and must be the mappings' target.

Usage (from the repo root):
  TARGET_NAME=1fefet1c/default python3 -m run_scripts.fefet1c_area --model vit-large
"""
import argparse
import glob
import json
import os

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import per_kind_accel_rows
from run_scripts.fefet1c_tiles import RESULTS_DIR, TILES_DIR
from targets.target_support import TARGET

AREA_DIR = os.path.join(RESULTS_DIR, "fefet1c_area")


def area_breakdown(tiles: dict, unit_rows: dict) -> dict:
    """{part: um^2} of one placed mapping (tiles: one fefet1c_tiles result)."""
    org = TARGET.organization
    arrays = tiles["arrays"]
    field = f"area_um2_{TARGET.node_nm}nm"
    parts = {f"array {name.removesuffix('_um2')}": arrays * um2 for name, um2 in vars(TARGET.array_area).items()}
    # arrays the placement added beyond the search's count (fefet1c_tiles.py step 2)
    parts["added arrays"] = tiles["added_arrays"] * sum(vars(TARGET.array_area).values())
    parts["PE input registers"] = tiles["pes"] * org.pe_input_register_area_um2
    parts["tile buses"] = tiles["tiles"] * org.tile_bus_area_um2
    parts["routers"] = tiles["routers"] * org.router_area_um2
    for sub_kind, count in tiles["units"]["total"].items():
        if sub_kind not in unit_rows:
            raise KeyError(f"no area row for accelerator unit {sub_kind!r}")
        parts[f"accelerator {sub_kind}"] = count * unit_rows[sub_kind][field]
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.model, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no tile results in {os.path.join(TILES_DIR, args.model)}")
    out_dir = os.path.join(AREA_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    rows_by_flags: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            tiles = json.load(f)
        with open(tiles["mapping"]) as f:
            mapping = json.load(f)
        if tiles["target"] != TARGET.name or mapping["target"] != TARGET.name:
            raise ValueError(f"{path} is for {tiles['target']}, not {TARGET.name}")
        flags_key = json.dumps(mapping["flags"], sort_keys=True)
        if flags_key not in rows_by_flags:
            rows_by_flags[flags_key] = per_kind_accel_rows(build_tiled_cim_block(mapping["model"], mapping["flags"]))
        parts = area_breakdown(tiles, rows_by_flags[flags_key])

        total = sum(parts.values())
        arrays_total = sum(v for k, v in parts.items() if k.startswith("array "))
        accel_total = sum(v for k, v in parts.items() if k.startswith("accelerator "))
        result = {
            "target": TARGET.name, "model": tiles["model"], "cap": tiles["cap"], "node_nm": TARGET.node_nm,
            "tiles_file": path, "arrays": tiles["arrays"], "tiles": tiles["tiles"],
            "total_mm2": total / 1e6,
            "added_arrays": tiles["added_arrays"],
            "groups_mm2": {"arrays": arrays_total / 1e6, "added arrays": parts["added arrays"] / 1e6,
                           "accelerator units": accel_total / 1e6,
                           "PE input registers": parts["PE input registers"] / 1e6,
                           "tile buses": parts["tile buses"] / 1e6, "routers": parts["routers"] / 1e6},
            "parts_mm2": {k: v / 1e6 for k, v in parts.items()},
            "parts_pct": {k: 100 * v / total for k, v in parts.items()},
        }
        out = os.path.join(out_dir, f"cap_{tiles['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        groups = ", ".join(f"{k} {100 * v * 1e6 / total:.1f}%" for k, v in result["groups_mm2"].items())
        print(f"cap {tiles['cap']}: {total / 1e6:,.1f} mm^2 ({groups}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
