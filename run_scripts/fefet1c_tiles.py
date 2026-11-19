"""Places a finished 1F1C mapping onto the chip's tiles and counts what the tiles hold.

Runs on the mapping files fefet1c_greedy_sweep.py writes
(results/fefet1c_greedy/<model>/cap_<cap>.json); it never maps or searches. The chip
organization (arrays per PE, PEs per tile, tiles per router, router flit width) comes
from TARGET.organization (targets/1fefet1c/default/organization.json, ISAAC
organization).

Placement:
  1. Weights. Every resident copy is packed into arrays the same way the search counts
     them (xdsl_hardware's best-fit column-strip packing, in the same order), keeping
     track of which arrays each copy lands in. Arrays are numbered in packing order;
     array a is in tile a // (arrays_per_pe * pes_per_tile). The packed count must equal
     the mapping's stationary array count, or the pass raises. An op with weights
     computes on the arrays holding them.
  2. Ops without weights (both operands dynamic) borrow arrays that are free for the
     op's whole run, in schedule order, from the tiles nearest the tile of their first
     operand (Manhattan distance on the mesh). A tile's busy arrays at a time are the
     arrays of the ops running in it then. The search's count lets an op's arrays change
     during its run; here it keeps the same arrays, so it can find fewer free than the
     search counted. Then the missing arrays go into tiles added after the mapping's
     (added_arrays: the peak busy arrays of the added tiles), and are reported apart.
     A matvec whose dynamic rhs is staged (the mapping's volatile.rhs_hold_*,
     xdsl_hardware.OperationMapping) first gets its staging copy the same way, from
     the tiles nearest the rhs's producer, busy from the rhs's first batch to the
     matvec's start; its own arrays are busy from its setup's start (live_start_ns).
  3. Accelerator ops run in the tiles of their first dynamic operand's producer. An op
     over several tiles splits each unit count over them in proportion to the
     producer's arrays per tile (largest remainder, so each op's count is kept). Each
     tile provisions its own units, with the same peak-over-time and tree-borrowing
     rules as the chip-wide count (xdsl_accel_concurrency._provisioned_counts).
  4. Traffic. Every dynamic edge (producer -> consumer; weights are loaded once before
     inference and not counted) moves elements x bits. Its hops are the Manhattan
     distance between the producer's and the consumer's weighted centroid tiles (0 when
     both sit in the same tiles). It crosses the tile bus once, or twice (both ends) when
     it leaves the tile, and the routers as ceil(bits / flit) flits per hop. A staged
     rhs moves twice: producer -> staging copy, then staging copy -> the matvec's
     arrays (its setup writes the copies).
No buffer is placed: operands stay in the WMU arrays, and after a WMU read each
column's value waits in its ADC's output register until a shared lane takes it; each
lane has its own small register (counted in the lane's area).
Model inputs (block arguments) enter at tile 0. Tiles sit row-major on a near-square
mesh; routers = ceil(tiles / tiles_per_router).

Writes results/fefet1c_tiles/<model>/cap_<cap>.json.

The target is TARGET_NAME (environment variable, required; targets.target_support reads
it at import time), and must be the mappings' target.

Usage (from the repo root):
  TARGET_NAME=1fefet1c/default python3 -m run_scripts.fefet1c_tiles --model vit-large
"""
import argparse
import bisect
import glob
import json
import math
import os
from collections import deque

from compiler.xdsl_accel_concurrency import _provisioned_counts
from targets.target_support import TARGET

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
MAPPINGS_DIR = os.path.join(RESULTS_DIR, "fefet1c_greedy")   # fefet1c_greedy_sweep.py's mappings
TILES_DIR = os.path.join(RESULTS_DIR, "fefet1c_tiles")

# ops that only reinterpret or replicate their first input: located where it is
_PASSTHROUGH_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape",
                    # layout ops too: the timing model passes their operand's timing on
                    # (xdsl_timing: shape ops); unfold is a conv's im2col
                    "cim.transpose", "cim.unfold"}
# ops that produce the resident weights
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}


class _Packer:
    """xdsl_hardware.HardwareMapping._bin_pack_stationary_arrays_uncached, also
    returning the arrays (as (first, count) ranges) each placement lands in. Free space
    is kept as free columns -> ranges of arrays."""

    def __init__(self, capacity: int):
        self.capacity = capacity
        self.next_array = 0
        self.keys: list[int] = []
        self.pool: dict[int, deque] = {}

    def place(self, full_arrays: int, size: int, tiles: int) -> list[tuple[int, int]]:
        ranges = []
        if full_arrays:
            ranges.append((self.next_array, full_arrays))
            self.next_array += full_arrays
        own_new: dict[int, list] = {}
        idx = bisect.bisect_left(self.keys, size)
        remaining = tiles
        upto = idx
        while remaining > 0 and upto < len(self.keys):
            key = self.keys[upto]
            free = self.pool[key]
            while remaining > 0 and free:
                first, count = free[0]
                take = min(count, remaining)
                ranges.append((first, take))
                remaining -= take
                if key - size > 0:
                    own_new.setdefault(key - size, []).append((first, take))
                if take == count:
                    free.popleft()
                else:
                    free[0] = (first + take, count - take)
            if not free:
                del self.pool[key]
                upto += 1
        if upto > idx:
            del self.keys[idx:upto]
        if remaining > 0:
            ranges.append((self.next_array, remaining))
            own_new.setdefault(self.capacity - size, []).append((self.next_array, remaining))
            self.next_array += remaining
        for key, new in own_new.items():
            if key in self.pool:
                self.pool[key].extend(new)
            else:
                self.pool[key] = deque(new)
                bisect.insort(self.keys, key)
        return ranges


def _arrays_per_tile(ranges, arrays_per_tile: int) -> dict[int, int]:
    per_tile: dict[int, int] = {}
    for first, count in ranges:
        a, end = first, first + count
        while a < end:
            tile = a // arrays_per_tile
            stop = min(end, (tile + 1) * arrays_per_tile)
            per_tile[tile] = per_tile.get(tile, 0) + stop - a
            a = stop
    return per_tile


def _apportion(count: int, weights: dict[int, int]) -> dict[int, int]:
    """count split over the tiles in proportion to weights, by largest remainder."""
    total = sum(weights.values())
    quotas = {t: count * w / total for t, w in weights.items()}
    shares = {t: int(q) for t, q in quotas.items()}
    left = count - sum(shares.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - shares[t]), t))[:left]:
        shares[t] += 1
    return {t: s for t, s in shares.items() if s}


def _max_busy(intervals, start: float, end: float) -> int:
    """Peak of the counts of intervals overlapping [start, end)."""
    events = []
    for s, e, c in intervals:
        if s < end and e > start:
            events.append((max(s, start), 1, c))
            events.append((min(e, end), 0, -c))
    busy = peak = 0
    for _, _, delta in sorted(events):
        busy += delta
        peak = max(peak, busy)
    return peak


def place_mapping(mapping: dict) -> dict:
    org = TARGET.organization
    arrays_per_tile = org.arrays_per_pe * org.pes_per_tile
    total_arrays = mapping["arrays"]["total"]
    n_tiles = -(-total_arrays // arrays_per_tile)
    width = math.ceil(math.sqrt(n_tiles))

    def coords(tile):
        return tile % width, tile // width

    mapped_tiles = n_tiles

    def capacity(tile):
        if tile >= mapped_tiles:   # an added tile
            return arrays_per_tile
        return min(arrays_per_tile, total_arrays - tile * arrays_per_tile)

    ops = {o["id"]: o for o in mapping["ops"]}
    arg_ids = {a["id"] for a in mapping["args"]}

    # 1. weights: pack every (stationary value, consumer) pair in the search's order
    pairs = []
    for o in mapping["ops"]:
        for s in o.get("stationary", []):
            pairs.append((s["value"], o["attributes"]["fx_name"], o, s))
    pairs.sort(key=lambda p: (p[0], p[1]))
    packer = _Packer(TARGET.geometry.rmu.cols)
    home: dict[int, dict[int, int]] = {}
    for _, _, o, s in pairs:
        ranges = packer.place(s["full_arrays_per_copy"] * s["copies"], s["partial_tile_cols"],
                              s["partial_tiles_per_copy"] * s["copies"])
        tiles = home.setdefault(o["id"], {})
        for tile, n in _arrays_per_tile(ranges, arrays_per_tile).items():
            tiles[tile] = tiles.get(tile, 0) + n
    if packer.next_array != mapping["arrays"]["nonvolatile"]:
        raise RuntimeError(f"weight packing gives {packer.next_array} arrays, the mapping counts "
                           f"{mapping['arrays']['nonvolatile']}")

    # arrays each tile has busy over time: ops with weights compute on their own arrays
    busy: dict[int, list] = {}
    for oid, tiles in home.items():
        o = ops[oid]
        if sum(tiles.values()) != o["volatile"]["arrays"]:
            raise RuntimeError(f"op {oid} ({o['op']}): {sum(tiles.values())} weight arrays but "
                               f"{o['volatile']['arrays']} working arrays")
        for tile, n in tiles.items():
            busy.setdefault(tile, []).append((o["volatile"]["live_start_ns"], o["volatile"]["live_end_ns"], n))

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

    def borrow(near: dict[int, int], start: float, end: float, need: int) -> dict[int, int]:
        """need arrays free over [start, end), from the tiles nearest the heaviest tile of
        `near`, adding tiles when none has room; marks them busy."""
        nonlocal n_tiles
        anchor = coords(max(near, key=lambda t: (near[t], -t)))
        placed: dict[int, int] = {}
        for tile in sorted(range(n_tiles), key=lambda t: (abs(coords(t)[0] - anchor[0])
                                                           + abs(coords(t)[1] - anchor[1]), t)):
            free = capacity(tile) - _max_busy(busy.get(tile, []), start, end)
            if free <= 0:
                continue
            take = min(free, need)
            placed[tile] = take
            busy.setdefault(tile, []).append((start, end, take))
            need -= take
            if need == 0:
                break
        while need:
            # no tile has room for the whole run: add a tile
            tile = n_tiles
            n_tiles += 1
            take = min(arrays_per_tile, need)
            placed[tile] = take
            busy.setdefault(tile, []).append((start, end, take))
            need -= take
        return placed

    units_by_tile: dict[int, dict[str, list]] = {}
    staged: dict[int, dict[int, int]] = {}  # matvec id -> its staging copy's {tile: arrays}
    # ops in block order (ids are node positions), so every producer is placed first
    for o in sorted(mapping["ops"], key=lambda o: o["id"]):
        oid = o["id"]
        kind = o["op"]
        if kind in _WEIGHT_OPS or kind in _PASSTHROUGH_OPS or oid in home:
            continue
        if "volatile" in o:
            # 2. an op without weights: its staged rhs nearest the rhs's producer, then
            # its own arrays nearest its first operand
            vol = o["volatile"]
            if vol.get("rhs_hold_arrays"):
                staged[oid] = borrow(located(o["inputs"][1]["from"]), vol["rhs_hold_start_ns"],
                                     vol["rhs_hold_end_ns"], vol["rhs_hold_arrays"])
            home[oid] = borrow(located(first_dynamic_input(o)), vol["live_start_ns"], vol["live_end_ns"],
                               vol["arrays"])
            continue
        if "units" not in o:
            raise NotImplementedError(f"op {oid} ({kind}): no placement rule")
        # 3. accelerator ops: in their producer's tiles
        weights = located(first_dynamic_input(o))
        home[oid] = weights
        start, end = o["schedule"]["start_ns"], o["schedule"]["end_ns"]
        for sub_kind, count in o["units"].items():
            for tile, n in _apportion(count, weights).items():
                units_by_tile.setdefault(tile, {}).setdefault(sub_kind, []).append((start, end, n))

    # 4. traffic of every dynamic edge (weights are loaded once, before inference)
    def centroid(tiles: dict[int, int]) -> tuple[float, float]:
        total = sum(tiles.values())
        return (sum(coords(t)[0] * w for t, w in tiles.items()) / total,
                sum(coords(t)[1] * w for t, w in tiles.items()) / total)

    edge_bits = bus_bits = flit_hops = 0

    def move(bits: int, src_tiles: dict[int, int], dst_tiles: dict[int, int]) -> None:
        nonlocal edge_bits, bus_bits, flit_hops
        producer, consumer = centroid(src_tiles), centroid(dst_tiles)
        hops = round(abs(producer[0] - consumer[0]) + abs(producer[1] - consumer[1]))
        edge_bits += bits
        bus_bits += bits * (2 if hops else 1)
        flit_hops += -(-bits // org.router_flit_bits) * hops

    for o in mapping["ops"]:
        if o["op"] in _WEIGHT_OPS or o["op"] in _PASSTHROUGH_OPS:
            continue
        for pos, inp in enumerate(o["inputs"]):
            src = inp["from"]
            if src not in arg_ids and ops[src]["op"] in _WEIGHT_OPS:
                continue
            bits = inp["elements"] * inp["bits"]
            if pos == 1 and o["id"] in staged:
                # a staged rhs: into its staging copy, then into the matvec's copies
                move(bits, located(src), staged[o["id"]])
                move(bits, staged[o["id"]], home[o["id"]])
            else:
                move(bits, located(src), home[o["id"]])

    units_total: dict[str, int] = {}
    units_max_tile: dict[str, int] = {}
    for tile, intervals in units_by_tile.items():
        for sub_kind, n in _provisioned_counts(intervals).items():
            units_total[sub_kind] = units_total.get(sub_kind, 0) + n
            units_max_tile[sub_kind] = max(units_max_tile.get(sub_kind, 0), n)

    added_arrays = sum(_max_busy(busy[t], -math.inf, math.inf) for t in range(mapped_tiles, n_tiles))
    return {
        "arrays": total_arrays,
        "added_arrays": added_arrays,
        "staged_arrays": sum(sum(t.values()) for t in staged.values()),
        "arrays_per_tile": arrays_per_tile,
        "tiles": n_tiles,
        "added_tiles": n_tiles - mapped_tiles,
        "mesh": [width, -(-n_tiles // width)],
        "pes": -(-(total_arrays + added_arrays) // org.arrays_per_pe),
        "routers": -(-n_tiles // org.tiles_per_router),
        "units": {"total": units_total, "max_per_tile": units_max_tile,
                  "chip_pool": {k: v for k, v in mapping["accelerators"].items() if v}},
        "traffic": {"edge_bits": edge_bits, "bus_bits": bus_bits, "flit_hops": flit_hops},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(MAPPINGS_DIR, args.model, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no mappings in {os.path.join(MAPPINGS_DIR, args.model)}")
    out_dir = os.path.join(TILES_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)
    for path in paths:
        with open(path) as f:
            mapping = json.load(f)
        if mapping["target"] != TARGET.name:
            raise ValueError(f"{path} is for {mapping['target']}, not {TARGET.name}")
        result = {"target": TARGET.name, "model": mapping["model"], "cap": mapping["cap"],
                  "mapping": path, "latency_ns": mapping["latency_ns"], **place_mapping(mapping)}
        out = os.path.join(out_dir, f"cap_{mapping['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"cap {mapping['cap']}: {result['tiles']} tiles, {result['routers']} routers, "
              f"{result['added_arrays']} added arrays -> {out}", flush=True)


if __name__ == "__main__":
    main()
