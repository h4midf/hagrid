"""Chip area of a placed HyFlexPIM mapping, per part.

Runs on the files hyflexpim_tiles.py writes (results/hyflexpim_tiles/<model>/
cap_<cap>.json); it never maps, searches or places. Areas are at the target's node
(TARGET.node_nm, 65 nm):

  analog modules     modules x each component of one analog module (TARGET.analog.
                     components: our Table 2 reproduction, targets/hyflexpim/table2.py)
  digital modules    modules x each component of one digital module, without the SFU:
                     our CMOS units run the non-linear ops instead
  inner buses        PUs x TARGET.inner_bus (ISAAC's tile bus at 1000 GB/s)
  routers            ceil(PUs / pus_per_router) x TARGET.global_router (ISAAC's router at
                     128 GB/s): the global bus
  accelerator units  per-PU unit counts x each unit's area (with its lane register), from
                     xdsl_accel_concurrency.per_kind_accel_rows of the model's block,
                     rebuilt with the mapping's own flags, x TARGET.cmos_area_factor

Whole modules are counted, as the placement fills them. Not counted: off-chip I/O and
chip-level control, as for 1F1C. Writes results/hyflexpim_area/<model>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.hyflexpim_area --model bert-base
"""
import argparse
import glob
import json
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "hyflexpim/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import per_kind_accel_rows
from run_scripts.hyflexpim_greedy_sweep import RESULTS_DIR
from run_scripts.hyflexpim_tiles import TILES_DIR
from targets.target_support import TARGET

AREA_DIR = os.path.join(RESULTS_DIR, "hyflexpim_area")
EXCLUDED_DIGITAL = ("sfu",)   # replaced by our CMOS units


def area_breakdown(tiles: dict, unit_rows: dict) -> dict:
    """{part: um^2} of one placed mapping (tiles: one hyflexpim_tiles result)."""
    parts = {}
    for name, c in TARGET.analog.components.items():
        parts[f"analog {name}"] = tiles["analog_modules"] * c.area_um2
    for name, c in TARGET.digital.components.items():
        if name not in EXCLUDED_DIGITAL:
            parts[f"digital {name}"] = tiles["digital_modules"] * c.area_um2
    parts["inner buses"] = tiles["pus"] * TARGET.inner_bus.area_um2
    parts["routers"] = -(-tiles["pus"] // TARGET.pus_per_router) * TARGET.global_router.area_um2
    for sub_kind, count in tiles["units"]["total"].items():
        if sub_kind not in unit_rows:
            raise KeyError(f"no area row for accelerator unit {sub_kind!r}")
        parts[f"accelerator {sub_kind}"] = count * unit_rows[sub_kind]["area_um2_45nm"] * TARGET.cmos_area_factor
    return parts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.model, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no placements in {os.path.join(TILES_DIR, args.model)}")
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
        groups = {g: sum(v for k, v in parts.items() if k.startswith(prefix)) / 1e6
                  for g, prefix in (("analog modules", "analog "), ("digital modules", "digital "),
                                    ("accelerator units", "accelerator "), ("inner buses", "inner buses"),
                                    ("routers", "routers"))}
        result = {
            "target": TARGET.name, "model": tiles["model"], "cap": tiles["cap"], "node_nm": TARGET.node_nm,
            "tiles_file": path, "analog_modules": tiles["analog_modules"],
            "digital_modules": tiles["digital_modules"], "pus": tiles["pus"],
            "total_mm2": total / 1e6, "groups_mm2": groups,
            "parts_mm2": {k: v / 1e6 for k, v in parts.items()},
            "parts_pct": {k: 100 * v / total for k, v in parts.items()},
        }
        out = os.path.join(out_dir, f"cap_{tiles['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        shares = ", ".join(f"{k} {100 * v * 1e6 / total:.1f}%" for k, v in groups.items())
        print(f"cap {tiles['cap']}: {total / 1e6:,.1f} mm^2 ({shares}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
