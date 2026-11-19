"""Chip area of a placed RRAM/SRAM ensemble mapping, per part.

Runs on the tile files rse_tiles.py writes (results/rse_tiles/<app>/cap_<cap>.json); it
never maps, searches or places. Areas are at the target's node (TARGET.node_nm):

  RRAM macros         RRAM macros x TARGET.costs.rram_macro_area_um2 (crossbar, flash
                      ADCs, headers, muxes, WL drivers and level shifters, decoder)
  SRAM macros         SRAM macros x TARGET.costs.sram_macro_area_um2 (array, PEs, FSM,
                      input and output buffers)
  added macros        SRAM macros the placement added beyond the search's count
                      (rse_tiles.py step 2), x the same SRAM macro area
  PE input registers  PEs x TARGET.organization.pe_input_register_area_um2
  tile buses          tiles x TARGET.organization.tile_bus_area_um2
  routers             routers x TARGET.organization.router_area_um2
  accelerator units   per-tile unit counts x each unit's synthesized area (with its lane
                      register), from xdsl_accel_concurrency.per_kind_accel_rows of the
                      app's block, rebuilt with the mapping's own flags, x
                      TARGET.cmos_area_factor (NanGate45 -> 65 nm)

Not counted (organization.json "excluded"): off-chip I/O and chip-level control.
Writes results/rse_area/<app>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.rse_area --app resnet50
"""
import argparse
import glob
import json
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import per_kind_accel_rows
from run_scripts.rse_greedy_sweep import RESULTS_DIR
from run_scripts.rse_tiles import TILES_DIR
from targets.target_support import TARGET

AREA_DIR = os.path.join(RESULTS_DIR, "rse_area")


def area_breakdown(tiles: dict, unit_rows: dict) -> dict:
    """{part: um^2} of one placed mapping (tiles: one rse_tiles result)."""
    org, costs = TARGET.organization, TARGET.costs
    parts = {
        "RRAM macros": tiles["rram_macros"] * costs.rram_macro_area_um2,
        "SRAM macros": tiles["sram_macros"] * costs.sram_macro_area_um2,
        "added macros": tiles["added_macros"] * costs.sram_macro_area_um2,
        "PE input registers": tiles["pes"] * org.pe_input_register_area_um2,
        "tile buses": tiles["tiles"] * org.tile_bus_area_um2,
        "routers": tiles["routers"] * org.router_area_um2,
    }
    for sub_kind, count in tiles["units"]["total"].items():
        if sub_kind not in unit_rows:
            raise KeyError(f"no area row for accelerator unit {sub_kind!r}")
        parts[f"accelerator {sub_kind}"] = count * unit_rows[sub_kind]["area_um2_45nm"] * TARGET.cmos_area_factor
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.app, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no tile results in {os.path.join(TILES_DIR, args.app)}")
    out_dir = os.path.join(AREA_DIR, args.app)
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
        accel_total = sum(v for k, v in parts.items() if k.startswith("accelerator "))
        groups = {k: parts[k] for k in ("RRAM macros", "SRAM macros", "added macros", "PE input registers",
                                        "tile buses", "routers")}
        groups["accelerator units"] = accel_total
        result = {
            "target": TARGET.name, "app": tiles["app"], "model": tiles["model"], "cap": tiles["cap"],
            "node_nm": TARGET.node_nm, "tiles_file": path, "macros": tiles["macros"],
            "rram_macros": tiles["rram_macros"], "sram_macros": tiles["sram_macros"],
            "added_macros": tiles["added_macros"], "tiles": tiles["tiles"],
            "total_mm2": total / 1e6,
            "groups_mm2": {k: v / 1e6 for k, v in groups.items()},
            "parts_mm2": {k: v / 1e6 for k, v in parts.items()},
            "parts_pct": {k: 100 * v / total for k, v in parts.items()},
        }
        out = os.path.join(out_dir, f"cap_{tiles['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        shares = ", ".join(f"{k} {100 * v / total:.1f}%" for k, v in groups.items())
        print(f"cap {tiles['cap']}: {total / 1e6:,.1f} mm^2 ({shares}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
