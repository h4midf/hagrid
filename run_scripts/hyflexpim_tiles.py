"""Places a finished HyFlexPIM mapping onto the chip's modules and processing units (PUs)
and counts what they hold and the data they exchange.

Runs on the mapping files hyflexpim_greedy_sweep.py writes
(results/hyflexpim_greedy/<model>/cap_<cap>.json); it never maps or searches. The chip
organization comes from TARGET (targets/hyflexpim/default): analog modules of
TARGET.analog.arrays arrays, digital modules of TARGET.digital.arrays arrays, and PUs of
analog_modules_per_pu analog + digital_modules_per_pu digital modules on one inner bus;
PUs talk over the global bus.

Placement:
  1. Weights. Every resident copy is packed into analog arrays the same way the search
     counts them (xdsl_hardware's best-fit column-strip packing, in the same order), so
     the packed count must equal the mapping's analog stationary count, or the pass
     raises. Analog array a is in analog module a // TARGET.analog.arrays, which is in PU
     module // analog_modules_per_pu. An op with weights computes on their arrays.
  2. Digital matvecs (both operands dynamic) borrow digital arrays free for the op's
     whole run: first in the PU holding most of their first operand, then in the next
     PUs in order. A PU holds digital_modules_per_pu x TARGET.digital.arrays digital
     arrays; the chip has as many as the mapping's digital count. Raises if no room.
     Before its compute arrays, a matvec's stored rhs (the mapping's volatile.rhs_hold_*,
     xdsl_dual_module_hardware) borrows the same way from the PU holding most of the
     rhs, busy from the rhs's first batch to the matvec's end.
  3. Accelerator ops (our CMOS units, in place of the paper's SFU) run where their first
     dynamic operand is, each PU provisioning its own units with the chip-wide count's
     peak-over-time and tree-borrowing rules (xdsl_accel_concurrency._provisioned_counts).
  4. Traffic. Every dynamic edge (weights are loaded once, before inference) moves
     elements x bits. The share of its producer and consumer in the same location stays
     local; the rest crosses the inner bus if it stays in its PU, or the global bus if it
     leaves it (from the overlap of the producer's and consumer's array distributions).
     Locations are analog modules, and each PU's digital modules taken together. A
     stored rhs moves twice: producer -> its stored copy -> the matvec's compute arrays.
Model inputs (block arguments) enter at analog module 0. PUs = the larger of the analog
and digital modules' PU counts.

Writes results/hyflexpim_tiles/<model>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.hyflexpim_tiles --model bert-base
"""
import argparse
import bisect
import glob
import json
import os
from collections import deque

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "hyflexpim/default"

from compiler.xdsl_accel_concurrency import _provisioned_counts
from run_scripts.hyflexpim_greedy_sweep import MAPPINGS_DIR, RESULTS_DIR
from targets.target_support import TARGET

TILES_DIR = os.path.join(RESULTS_DIR, "hyflexpim_tiles")

# ops that only reinterpret or replicate their first input: located where it is
_PASSTHROUGH_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape", "cim.transpose", "cim.unfold"}
# ops that produce the resident weights
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}


class _Packer:
    """The search's best-fit column-strip packing (as run_scripts/fefet1c_tiles.py's), also
    returning the arrays (as (first, count) ranges) each placement lands in."""

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


def _per_module(ranges, arrays_per_module: int) -> dict[int, int]:
    out: dict[int, int] = {}
    for first, count in ranges:
        a, end = first, first + count
        while a < end:
            m = a // arrays_per_module
            stop = min(end, (m + 1) * arrays_per_module)
            out[m] = out.get(m, 0) + stop - a
            a = stop
    return out


def _apportion(count: int, weights: dict) -> dict:
    """count split in proportion to weights, by largest remainder."""
    total = sum(weights.values())
    quotas = {t: count * w / total for t, w in weights.items()}
    shares = {t: int(q) for t, q in quotas.items()}
    left = count - sum(shares.values())
    for t in sorted(quotas, key=lambda t: (-(quotas[t] - shares[t]), str(t)))[:left]:
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


def _overlap(a: dict, b: dict) -> float:
    """Shared fraction of two location distributions."""
    ta, tb = sum(a.values()), sum(b.values())
    return sum(min(w / ta, b[k] / tb) for k, w in a.items() if k in b)


def place_mapping(mapping: dict) -> dict:
    an, dg = TARGET.analog, TARGET.digital
    a_per_pu, d_per_pu = TARGET.analog_modules_per_pu, TARGET.digital_modules_per_pu
    analog_total = mapping["arrays"]["analog"]["total"]
    digital_total = mapping["arrays"]["digital"]["total"]
    analog_modules = -(-analog_total // an.arrays)
    digital_modules = -(-digital_total // dg.arrays)
    pus = max(-(-analog_modules // a_per_pu), -(-digital_modules // d_per_pu))
    d_per_pu_arrays = d_per_pu * dg.arrays

    def digital_capacity(pu: int) -> int:
        return max(0, min(d_per_pu_arrays, digital_total - pu * d_per_pu_arrays))

    def pu_of(loc) -> int:
        kind, index = loc
        return index // a_per_pu if kind == "a" else index

    ops = {o["id"]: o for o in mapping["ops"]}
    arg_ids = {a["id"] for a in mapping["args"]}

    # 1. weights, in the search's order
    pairs = []
    for o in mapping["ops"]:
        for s in o.get("stationary", []):
            pairs.append((s["value"], o["attributes"]["fx_name"], o, s))
    pairs.sort(key=lambda p: (p[0], p[1]))
    packer = _Packer(an.array.cols)
    home: dict[int, dict] = {}
    for _, _, o, s in pairs:
        ranges = packer.place(s["full_arrays_per_copy"] * s["copies"], s["partial_tile_cols"],
                              s["partial_tiles_per_copy"] * s["copies"])
        locs = home.setdefault(o["id"], {})
        for m, n in _per_module(ranges, an.arrays).items():
            locs[("a", m)] = locs.get(("a", m), 0) + n
    if packer.next_array != mapping["arrays"]["analog"]["nonvolatile"]:
        raise RuntimeError(f"weight packing gives {packer.next_array} analog arrays, the mapping counts "
                           f"{mapping['arrays']['analog']['nonvolatile']}")
    for oid, locs in home.items():
        o = ops[oid]
        if sum(locs.values()) != o["volatile"]["arrays"]:
            raise RuntimeError(f"op {oid} ({o['op']}): {sum(locs.values())} weight arrays but "
                               f"{o['volatile']['arrays']} working arrays")

    def located(value_id) -> dict:
        if value_id in arg_ids:
            return {("a", 0): 1}
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

    digital_busy: dict[int, list] = {}

    def borrow(near: dict, start: float, end: float, need: int, what: str) -> dict:
        """need digital arrays free over [start, end), from the PU holding most of `near`
        on; marks them busy. Raises if no room."""
        by_pu: dict[int, int] = {}
        for loc, w in near.items():
            by_pu[pu_of(loc)] = by_pu.get(pu_of(loc), 0) + w
        anchor = max(by_pu, key=lambda p: (by_pu[p], -p))
        placed: dict = {}
        left = need
        for step in range(pus):
            pu = (anchor + step) % pus
            free = digital_capacity(pu) - _max_busy(digital_busy.get(pu, []), start, end)
            if free <= 0:
                continue
            take = min(free, left)
            placed[("d", pu)] = take
            digital_busy.setdefault(pu, []).append((start, end, take))
            left -= take
            if left == 0:
                break
        if left:
            raise RuntimeError(f"{what}: no free digital arrays for {left} of {need}")
        return placed

    units_by_pu: dict[int, dict[str, list]] = {}
    stored: dict[int, dict] = {}  # matvec id -> its stored rhs's {location: arrays}
    for o in sorted(mapping["ops"], key=lambda o: o["id"]):
        oid, kind = o["id"], o["op"]
        if kind in _WEIGHT_OPS or kind in _PASSTHROUGH_OPS or oid in home:
            continue
        if kind == "cim.MatvecOp":
            # 2. digital matvec: its stored rhs, the rhs's PU first; then its compute
            # arrays, its first operand's PU first
            vol = o["volatile"]
            if vol.get("rhs_hold_arrays"):
                stored[oid] = borrow(located(o["inputs"][1]["from"]), vol["rhs_hold_start_ns"],
                                     vol["rhs_hold_end_ns"], vol["rhs_hold_arrays"], f"op {oid}'s stored rhs")
            home[oid] = borrow(located(first_dynamic_input(o)), vol["live_start_ns"], vol["live_end_ns"],
                               vol["arrays"], f"op {oid}")
            continue
        if "volatile" in o:
            raise NotImplementedError(f"op {oid} ({kind}): arrays without weights, not a digital matvec")
        if "units" not in o:
            raise NotImplementedError(f"op {oid} ({kind}): no placement rule")
        # 3. accelerator ops: where their operand is, provisioned per PU
        locs = located(first_dynamic_input(o))
        home[oid] = locs
        by_pu = {}
        for loc, w in locs.items():
            by_pu[pu_of(loc)] = by_pu.get(pu_of(loc), 0) + w
        start, end = o["schedule"]["start_ns"], o["schedule"]["end_ns"]
        for sub_kind, count in o["units"].items():
            for pu, n in _apportion(count, by_pu).items():
                units_by_pu.setdefault(pu, {}).setdefault(sub_kind, []).append((start, end, n))

    # 4. traffic
    local_bits = inner_bits = global_bits = 0.0

    def by_pu_of(locs: dict) -> dict:
        out: dict[int, int] = {}
        for loc, w in locs.items():
            out[pu_of(loc)] = out.get(pu_of(loc), 0) + w
        return out

    def move(bits: float, producer: dict, consumer: dict) -> None:
        nonlocal local_bits, inner_bits, global_bits
        same_loc = _overlap(producer, consumer)
        same_pu = _overlap(by_pu_of(producer), by_pu_of(consumer))
        local_bits += bits * same_loc
        inner_bits += bits * (same_pu - same_loc)
        global_bits += bits * (1 - same_pu)

    for o in mapping["ops"]:
        if o["op"] in _WEIGHT_OPS or o["op"] in _PASSTHROUGH_OPS:
            continue
        for pos, inp in enumerate(o["inputs"]):
            src = inp["from"]
            if src not in arg_ids and ops[src]["op"] in _WEIGHT_OPS:
                continue
            bits = inp["elements"] * inp["bits"]
            if pos == 1 and o["id"] in stored:
                # a stored rhs: into its stored copy, then to the compute arrays
                move(bits, located(src), stored[o["id"]])
                move(bits, stored[o["id"]], home[o["id"]])
            else:
                move(bits, located(src), home[o["id"]])

    units_total: dict[str, int] = {}
    units_max_pu: dict[str, int] = {}
    for pu, intervals in units_by_pu.items():
        for sub_kind, n in _provisioned_counts(intervals).items():
            units_total[sub_kind] = units_total.get(sub_kind, 0) + n
            units_max_pu[sub_kind] = max(units_max_pu.get(sub_kind, 0), n)

    return {
        "analog_arrays": analog_total, "digital_arrays": digital_total,
        "analog_modules": analog_modules, "digital_modules": digital_modules, "pus": pus,
        "stored_rhs_arrays": sum(sum(s.values()) for s in stored.values()),
        "units": {"total": units_total, "max_per_pu": units_max_pu,
                  "chip_pool": {k: v for k, v in mapping["accelerators"].items() if v}},
        "traffic": {"local_bits": local_bits, "inner_bus_bits": inner_bits, "global_bus_bits": global_bits},
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
        result = {"target": TARGET.name, "model": mapping["model"], "cap": mapping["cap"],
                  "mapping": path, "latency_ns": mapping["latency_ns"], **place_mapping(mapping)}
        out = os.path.join(out_dir, f"cap_{mapping['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        t = result["traffic"]
        print(f"cap {mapping['cap']}: {result['analog_modules']} analog + {result['digital_modules']} digital "
              f"modules, {result['pus']} PUs; bus Mbit: inner {t['inner_bus_bits'] / 1e6:.1f}, "
              f"global {t['global_bus_bits'] / 1e6:.1f}, local {t['local_bits'] / 1e6:.1f} -> {out}", flush=True)


if __name__ == "__main__":
    main()
