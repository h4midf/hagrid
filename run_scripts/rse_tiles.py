"""Places a finished RRAM/SRAM ensemble mapping onto the chip's tiles and counts what the
tiles hold.

Runs on the mapping files rse_greedy_sweep.py writes (results/rse_greedy/<app>/
cap_<cap>.json); it never maps or searches. The chip organization (macro slots per PE,
PEs per tile, tiles per router, router flit width) comes from TARGET.organization
(targets/rram-sram-ensemble/default/organization.json, ISAAC organization, as 1F1C's).
Each of a PE's slots holds one macro of either kind.

Placement:
  1. Weight blocks. Every resident copy is packed into RRAM macros the same way the search
     counts them (xdsl_hardware's best-fit column-strip packing, in the same order:
     fefet1c_tiles _Packer); the packed count must equal the mapping's RRAM count, or the
     pass raises. Each tile gives its block share of slots (ceil(slots x (RRAM + SRAM
     correction) / all macros)) to blocks, handed out in packing order: each RRAM macro the
     packer opens, then its share of the op's SRAM correction macros (correction_plan, at
     most sram_macros_per_rram_macro per RRAM macro), so every macro's correction sits
     beside it, as the paper pairs each SRAM macro with an RRAM macro (Fig. 2). An op with
     weights computes on the macros of its block.
  2. Dynamic matvecs' SRAM macros, in the rest of each tile's slots, shared over time
     between them, nearest the op's first operand (Manhattan distance on the mesh), free
     for the op's whole run. The search's count lets an op's macros change during its run;
     here it keeps the same ones, so it can find fewer free than the search counted. Then
     the missing macros go into tiles added after the mapping's (added_macros), reported
     apart. Before them, a dynamic matvec's staged rhs (the mapping's volatile.rhs_hold_*,
     xdsl_ensemble_hardware) takes SRAM slots the same way nearest the rhs's producer,
     from the rhs's first batch to the matvec's start; the matvec's own macros are busy
     from its setup's start (live_start_ns).
  3. Accelerator ops run in the tiles of their first dynamic operand's producer, each
     tile provisioning its own units (fefet1c_tiles step 3).
  4. Traffic. Every dynamic edge moves elements x bits from the producer's to the
     consumer's centroid tile once (a dynamic matvec's rhs multicast to its pf copies; a
     staged rhs first into its staging copy, then from it to the copies), plus,
     per stored matvec, its correction's outputs (every output element, accumulator bits,
     from the SRAM macros to the RRAM macros, where the shifter adds them). Each crosses
     the tile bus once, or twice when it leaves the tile, and the routers as ceil(bits /
     flit) flits per hop.
No buffer is placed: activations live in the SRAM macros (the WMU), as 1F1C's in its WMU.
Model inputs (block arguments) enter at tile 0. Tiles sit row-major on a near-square
mesh; routers = ceil(tiles / tiles_per_router).

Writes results/rse_tiles/<app>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.rse_tiles --app resnet50
"""
import argparse
import glob
import json
import math
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from compiler.xdsl_accel_concurrency import _provisioned_counts
from compiler.xdsl_ensemble_timing import correction_plan
from run_scripts.fefet1c_tiles import _apportion, _arrays_per_tile, _max_busy, _Packer
from run_scripts.rse_greedy_sweep import MAPPINGS_DIR, RESULTS_DIR
from targets.target_loader import ensemble_for_app
from targets.target_support import TARGET

TILES_DIR = os.path.join(RESULTS_DIR, "rse_tiles")

# ops that only reinterpret or replicate their first input: located where it is
_PASSTHROUGH_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape", "cim.transpose", "cim.unfold"}
# ops that produce the resident weights
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}
_FOREVER = (-math.inf, math.inf)


def stored_correction(target, o, ops):
    """The CorrectionPlan of a stored matvec, checked against the mapping's count."""
    (s,) = o["stationary"]
    weight = ops[s["value"]]
    plan = correction_plan(target, weight["shape"], weight["bits"], o["pf"])
    if plan.macros != o["sram_correction_macros"]:
        raise RuntimeError(f"op {o['id']}: correction_plan gives {plan.macros} SRAM correction macros, the "
                           f"mapping {o['sram_correction_macros']}")
    return plan


def place_mapping(mapping: dict) -> dict:
    target = ensemble_for_app(TARGET, mapping["app"])
    org = TARGET.organization
    slots_per_tile = org.arrays_per_pe * org.pes_per_tile
    counts = mapping["arrays"]
    total = counts["total"]
    ops = {o["id"]: o for o in mapping["ops"]}
    arg_ids = {a["id"] for a in mapping["args"]}
    stored = [o for o in mapping["ops"] if o["op"] == "cim.MatvecOp" and "stationary" in o]
    correction = {o["id"]: stored_correction(target, o, ops) for o in stored}
    if sum(c.macros for c in correction.values()) != counts["sram"]["nonvolatile"]:
        raise RuntimeError(f"{sum(c.macros for c in correction.values())} SRAM correction macros, the mapping "
                           f"counts {counts['sram']['nonvolatile']}")

    # 1. weight blocks: each op's RRAM macros (in packing order), each followed by its share of
    # the op's SRAM correction macros; every tile gives block_per_tile slots to blocks, the
    # rest to the SRAM used while ops run (step 2)
    block_total = counts["rram"]["nonvolatile"] + counts["sram"]["nonvolatile"]
    block_per_tile = -(-slots_per_tile * block_total // total)
    pairs = []
    for o in mapping["ops"]:
        for s in o.get("stationary", []):
            pairs.append((s["value"], o["attributes"]["fx_name"], o, s))
    pairs.sort(key=lambda p: (p[0], p[1]))
    packer = _Packer(TARGET.rram.array.cols)
    rram_slot: list[int] = []      # packer macro -> slot
    slot = 0
    home: dict[int, dict[int, int]] = {}
    sram_home: dict[int, dict[int, int]] = {}

    def add(tiles: dict[int, int], s_: int):
        t = s_ // block_per_tile
        tiles[t] = tiles.get(t, 0) + 1

    for _, _, o, s in pairs:
        before = packer.next_array
        full = s["full_arrays_per_copy"] * s["copies"]
        ranges = packer.place(full, s["partial_tile_cols"], s["partial_tiles_per_copy"] * s["copies"])
        new = list(range(before, packer.next_array))
        sram_total = correction[o["id"]].macros if o["id"] in correction else 0
        sram_slots: list[int] = []
        # each new RRAM macro, then its share of the op's SRAM correction macros, so every
        # macro's correction sits beside it (copy by copy: its full macros, then its share
        # of the newly opened narrower macros)
        n_copies = s["copies"]
        partial_new = new[full:]
        order = []
        for k in range(n_copies):
            order += new[k * s["full_arrays_per_copy"]:(k + 1) * s["full_arrays_per_copy"]]
            order += partial_new[k * len(partial_new) // n_copies:(k + 1) * len(partial_new) // n_copies]
        rram_slot.extend([None] * len(new))
        for i, a in enumerate(order):
            rram_slot[a] = slot
            slot += 1
            share = (i + 1) * sram_total // len(order) - i * sram_total // len(order)
            sram_slots.extend(range(slot, slot + share))
            slot += share
        if not order:
            # every tile went into earlier macros' free columns: the SRAM follows
            sram_slots.extend(range(slot, slot + sram_total))
            slot += sram_total
        tiles = home.setdefault(o["id"], {})
        for first, count in ranges:
            for a in range(first, first + count):
                add(tiles, rram_slot[a])
        if sram_total:
            sram_home[o["id"]] = {}
            for s_ in sram_slots:
                add(sram_home[o["id"]], s_)
    if packer.next_array != counts["rram"]["nonvolatile"]:
        raise RuntimeError(f"RRAM packing gives {packer.next_array} macros, the mapping counts "
                           f"{counts['rram']['nonvolatile']}")
    for oid, tiles in home.items():
        if sum(tiles.values()) != ops[oid]["volatile"]["rram_macros"]:
            raise RuntimeError(f"op {oid}: {sum(tiles.values())} RRAM macros placed, "
                               f"{ops[oid]['volatile']['rram_macros']} in the mapping")
    block_slots = slot
    if block_slots != block_total:
        raise RuntimeError(f"{block_slots} block macros, the mapping counts {block_total}")

    # 2. the SRAM used while ops run, in the rest of each tile's slots, shared over time
    n_tiles = max(-(-block_slots // block_per_tile), -(-total // slots_per_tile))
    width = math.ceil(math.sqrt(n_tiles))

    def coords(tile):
        return tile % width, tile // width

    busy: dict[int, list] = {t: [(*_FOREVER, n)] for t, n in _arrays_per_tile([(0, block_slots)],
                                                                               block_per_tile).items()}

    def nearest(anchor_tiles: dict[int, int], need: int, start: float, end: float) -> dict[int, int]:
        """need SRAM slots free over [start, end), nearest the anchor tiles first; tiles are
        added when none is left."""
        nonlocal n_tiles
        anchor = coords(max(anchor_tiles, key=lambda t: (anchor_tiles[t], -t)))
        placed: dict[int, int] = {}
        for tile in sorted(range(n_tiles), key=lambda t: (abs(coords(t)[0] - anchor[0])
                                                           + abs(coords(t)[1] - anchor[1]), t)):
            free = slots_per_tile - _max_busy(busy.get(tile, []), start, end)
            if free <= 0:
                continue
            take = min(free, need)
            placed[tile] = take
            busy.setdefault(tile, []).append((start, end, take))
            need -= take
            if need == 0:
                return placed
        while need:
            tile = n_tiles
            n_tiles += 1
            take = min(slots_per_tile, need)
            placed[tile] = take
            busy.setdefault(tile, []).append((start, end, take))
            need -= take
        return placed

    def located(value_id) -> dict[int, int]:
        """Where a value is: {tile: weight}."""
        if value_id in arg_ids:
            return {0: 1}
        o = ops[value_id]
        if o["op"] in _PASSTHROUGH_OPS:
            return located(o["inputs"][0]["from"])
        if value_id not in home:
            raise RuntimeError(f"op {value_id} ({o['op']}) has no location yet")
        return home[value_id]

    def first_dynamic_input(o) -> int:
        for inp in o["inputs"]:
            src = inp["from"]
            if src in arg_ids or ops[src]["op"] not in _WEIGHT_OPS:
                return src
        raise RuntimeError(f"op {o['id']} ({o['op']}) has no dynamic input")

    units_by_tile: dict[int, dict[str, list]] = {}
    staged: dict[int, dict[int, int]] = {}  # dynamic matvec id -> its staged rhs's {tile: macros}
    # ops in block order (ids are node positions), so every producer is placed first
    for o in sorted(mapping["ops"], key=lambda o: o["id"]):
        oid, kind = o["id"], o["op"]
        if kind in _WEIGHT_OPS or kind in _PASSTHROUGH_OPS:
            continue
        if kind == "cim.MatvecOp":
            v = o["volatile"]
            if "stationary" in o:
                continue   # in its weight block (step 1)
            # 2. a dynamic matvec: its staged rhs near the rhs's producer, then its own
            # SRAM macros near its first operand
            if v.get("rhs_hold_arrays"):
                staged[oid] = nearest(located(o["inputs"][1]["from"]), v["rhs_hold_arrays"],
                                      v["rhs_hold_start_ns"], v["rhs_hold_end_ns"])
            home[oid] = nearest(located(first_dynamic_input(o)), v["sram_macros"],
                                v["live_start_ns"], v["live_end_ns"])
            continue
        if oid in home:
            continue
        if "volatile" in o:
            # a LUT lookup computes on the RRAM macros holding its table (step 1)
            raise RuntimeError(f"op {oid} ({kind}): a mapped op without a placed table")
        if "units" not in o:
            raise NotImplementedError(f"op {oid} ({kind}): no placement rule")
        # 3. accelerator ops: in their producer's tiles
        weights = located(first_dynamic_input(o))
        home[oid] = weights
        start, end = o["schedule"]["start_ns"], o["schedule"]["end_ns"]
        for sub_kind, count in o["units"].items():
            for tile, n in _apportion(count, weights).items():
                units_by_tile.setdefault(tile, {}).setdefault(sub_kind, []).append((start, end, n))

    # 4. traffic
    def centroid(tiles: dict[int, int]) -> tuple[float, float]:
        n = sum(tiles.values())
        return (sum(coords(t)[0] * w for t, w in tiles.items()) / n,
                sum(coords(t)[1] * w for t, w in tiles.items()) / n)

    traffic = {"edge_bits": 0, "bus_bits": 0, "flit_hops": 0, "correction_bits": 0,
               "flit_hops_by_kind": {"edge_bits": 0, "correction_bits": 0}}

    def move(bits: int, a: dict[int, int], b: dict[int, int], kind: str):
        pa, pb = centroid(a), centroid(b)
        hops = round(abs(pa[0] - pb[0]) + abs(pa[1] - pb[1]))
        flit_hops = -(-bits // org.router_flit_bits) * hops
        traffic[kind] += bits
        traffic["bus_bits"] += bits * (2 if hops else 1)
        traffic["flit_hops"] += flit_hops
        traffic["flit_hops_by_kind"][kind] += flit_hops

    for o in mapping["ops"]:
        if o["op"] in _WEIGHT_OPS or o["op"] in _PASSTHROUGH_OPS:
            continue
        for pos, inp in enumerate(o["inputs"]):
            src = inp["from"]
            if src not in arg_ids and ops[src]["op"] in _WEIGHT_OPS:
                continue
            bits = inp["elements"] * inp["bits"]
            if pos == 1 and o["id"] in staged:
                # a staged rhs: into its staging copy, then multicast to the pf copies
                move(bits, located(src), staged[o["id"]], "edge_bits")
                move(bits, staged[o["id"]], home[o["id"]], "edge_bits")
            else:
                # once per edge: a dynamic matvec's rhs is multicast to its pf copies
                move(bits, located(src), home[o["id"]], "edge_bits")
        if o["op"] == "cim.MatvecOp" and "stationary" in o:
            move(math.prod(o["shape"]) * TARGET.costs.sram_accumulator_bits, sram_home[o["id"]], home[o["id"]],
                 "correction_bits")

    units_total: dict[str, int] = {}
    units_max_tile: dict[str, int] = {}
    for tile, intervals in units_by_tile.items():
        for sub_kind, n in _provisioned_counts(intervals).items():
            units_total[sub_kind] = units_total.get(sub_kind, 0) + n
            units_max_tile[sub_kind] = max(units_max_tile.get(sub_kind, 0), n)

    # macros in use at the peak: the blocks (held for the whole run) and the dynamic
    # matvecs' peak; beyond the mapping's count, they are added (SRAM: the blocks' own copies)
    used = block_slots + _max_busy([iv for t in busy for iv in busy[t] if iv[0] != -math.inf], -math.inf, math.inf)
    added = max(0, used - total)
    return {
        "macros": total,
        "rram_macros": counts["rram"]["total"],
        "sram_macros": counts["sram"]["total"],
        "added_macros": added,
        "staged_macros": sum(sum(t.values()) for t in staged.values()),
        "slots_per_tile": slots_per_tile,
        "tiles": n_tiles,
        "added_tiles": max(0, n_tiles - -(-total // slots_per_tile)),
        "mesh": [width, -(-n_tiles // width)],
        "pes": -(-(total + added) // org.arrays_per_pe),
        "routers": -(-n_tiles // org.tiles_per_router),
        "units": {"total": units_total, "max_per_tile": units_max_tile,
                  "chip_pool": {k: v for k, v in mapping["accelerators"].items() if v}},
        "traffic": traffic,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(MAPPINGS_DIR, args.app, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no mappings in {os.path.join(MAPPINGS_DIR, args.app)}")
    out_dir = os.path.join(TILES_DIR, args.app)
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        with open(path) as f:
            mapping = json.load(f)
        result = {"target": TARGET.name, "app": mapping["app"], "model": mapping["model"], "cap": mapping["cap"],
                  "mapping": path, "latency_ns": mapping["latency_ns"], **place_mapping(mapping)}
        out = os.path.join(out_dir, f"cap_{mapping['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"cap {mapping['cap']}: {result['tiles']} tiles, {result['routers']} routers, "
              f"{result['added_macros']} added macros -> {out}", flush=True)


if __name__ == "__main__":
    main()
