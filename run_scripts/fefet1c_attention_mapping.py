"""The paper's mapping example (implementation, Fig. 6) on 1F1C: one int8 attention layer
(bert-attention-1layer, the first layer of the Fig. 3 app) under the shared offload policy
with softmax decomposed into lookups (sm_impl="lut"), mapped by the greedy search
(run_capped_search_sync_group) at CAP arrays, with a physical array placement of that
mapping.

The compiler gives array counts, not addresses: each stationary copy's arrays, and each
mapped op's working (WMU) arrays from its start to its end (a matvec with stationary
weights computes in the arrays that hold its copies; one with a dynamic rhs in WMUs it
writes the rhs into). The placement gives them addresses:

  stationary  every copy of every stationary operand, in block order, at consecutive
              addresses. Narrow LUT tiles receive dedicated arrays rather than
              sharing columns with independently scheduled consumers.
  working     an op with stationary weights at its copies' addresses; an op with a
              dynamic rhs at the lowest addresses no other op uses while it is live
              (from its setup's start), and its staged rhs copy ("held_addresses",
              from the rhs's first batch to the op's start) the same way, in order of
              start time

and checks it against the compiler: dedicated stationary placement fits within the
mapping's total arrays, peak working arrays equal the mapping's count, and no
working address is used by two ops at once. Column packing can make the compiler's
stationary count lower than this conservative dedicated placement.

Writes run_scripts/results/fefet1c_attention_mapping_bert-attention-1layer.json: the
mapping (compiler/xdsl_mapping_export.py) and the placement, address runs as [first, last].
plot/attention_mapping.py draws it.

Usage (from the repo root):
  python3 -m run_scripts.fefet1c_attention_mapping
"""
import json
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "1fefet1c/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_mapping_export import export_mapping
from compiler.xdsl_search import Budget, run_capped_search_sync_group
from run_scripts.offload_policy_common import offload_policy_flags
from targets.target_support import TARGET

APP = "bert-attention-1layer"
CAP = 2048  # the device of the paper's example
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
OUT_PATH = os.path.join(RESULTS_DIR, f"fefet1c_attention_mapping_{APP}.json")


def runs(addresses: list[int]) -> list[list[int]]:
    """Sorted addresses as [first, last] runs of consecutive ones."""
    out = []
    for a in sorted(addresses):
        if out and a == out[-1][1] + 1:
            out[-1][1] = a
        else:
            out.append([a, a])
    return out


def overlap(a: dict, b: dict) -> bool:
    """Closed live ranges overlap (as the compiler's peak count: a start at another's end counts)."""
    return a["live_start_ns"] <= b["live_end_ns"] and b["live_start_ns"] <= a["live_end_ns"]


def place(mapping: dict) -> dict:
    """The placement of `mapping` (export_mapping's dict), checked against its counts."""
    total = mapping["arrays"]["total"]
    stationary, by_consumer, cursor = [], {}, 0
    for op in mapping["ops"]:
        for s in op.get("stationary", []):
            # narrow LUT tiles get dedicated arrays here, so independently scheduled
            # lookups never share wordlines
            arrays_per_copy = s["full_arrays_per_copy"] + s["partial_tiles_per_copy"]
            count = s["copies"] * arrays_per_copy
            addresses = list(range(cursor, cursor + count))
            cursor += count
            stationary.append({"value": s["value"], "consumer": op["id"], "copies": s["copies"],
                               "arrays_per_copy": arrays_per_copy, "addresses": addresses})
            by_consumer.setdefault(op["id"], []).extend(addresses)
    if cursor < mapping["arrays"]["nonvolatile"] or cursor > total:
        raise RuntimeError(f"{cursor} dedicated stationary arrays do not fit the mapping's {total} arrays")

    working, pieces = [], []  # pieces: {"row", "live_start_ns", "live_end_ns", "addresses"}
    for op in mapping["ops"]:
        vol = op.get("volatile")
        if vol and vol["arrays"] > 0 and op["id"] in by_consumer:
            if len(by_consumer[op["id"]]) != vol["arrays"]:
                raise RuntimeError(f"op {op['id']}: {vol['arrays']} working arrays, {len(by_consumer[op['id']])} copies' arrays")
            row = {"op": op["id"], "rhs": "stationary", **vol, "addresses": by_consumer[op["id"]]}
            working.append(row)
            pieces.append({"row": row, "live_start_ns": vol["live_start_ns"], "live_end_ns": vol["live_end_ns"],
                           "addresses": row["addresses"]})
    # a dynamic-rhs op: its staged rhs copy (rhs's first batch to the op's start) and its
    # own arrays (its setup's start to its end; xdsl_hardware.OperationMapping), each
    # placed in time order
    todo = []
    for op in mapping["ops"]:
        vol = op.get("volatile")
        if not vol or vol["arrays"] <= 0 or op["id"] in by_consumer:
            continue
        row = {"op": op["id"], "rhs": "dynamic", "rhs_value": op["inputs"][1]["from"], **vol, "addresses": []}
        working.append(row)
        if vol.get("rhs_hold_arrays", 0):
            todo.append((vol["rhs_hold_start_ns"], vol["rhs_hold_end_ns"], op["id"], "staged", row,
                         vol["rhs_hold_arrays"]))
        todo.append((vol["live_start_ns"], vol["live_end_ns"], op["id"], "op", row, vol["arrays"]))
    for start, end, _id, part, row, count in sorted(todo, key=lambda x: x[:3]):
        piece = {"row": row, "live_start_ns": start, "live_end_ns": end}
        busy = {a for p in pieces if overlap(p, piece) for a in p["addresses"]}
        free = [a for a in range(total) if a not in busy]
        if len(free) < count:
            raise RuntimeError(f"op {row['op']}: needs {count} arrays at {start} ns, {len(free)} free of {total}")
        piece["addresses"] = free[:count]
        pieces.append(piece)
        if part == "staged":
            row["held_addresses"] = piece["addresses"]
        else:
            row["addresses"] = piece["addresses"]

    # no address in two pieces at once; the peak equals the compiler's
    for i, a in enumerate(pieces):
        for b in pieces[i + 1:]:
            if overlap(a, b) and set(a["addresses"]) & set(b["addresses"]):
                raise RuntimeError(f"ops {a['row']['op']} and {b['row']['op']} share arrays at the same time")
    # the peak is reached at some piece's start
    starts = sorted({p["live_start_ns"] for p in pieces})
    peak = max(sum(len(p["addresses"]) for p in pieces if p["live_start_ns"] <= t <= p["live_end_ns"]) for t in starts)
    if peak != mapping["arrays"]["volatile"]:
        raise RuntimeError(f"placed working peak {peak}, the mapping's is {mapping['arrays']['volatile']}")
    if max(a for p in pieces for a in p["addresses"]) >= total:
        raise RuntimeError("placement goes beyond the mapping's total arrays")
    for row in stationary + working:
        row["addresses"] = runs(row["addresses"])
        if "held_addresses" in row:
            row["held_addresses"] = runs(row["held_addresses"])
    return {"stationary": stationary, "working": working}


def main() -> None:
    flags = offload_policy_flags()
    flags["sm_impl"] = "lut"  # the paper's example decomposes softmax
    block = build_tiled_cim_block(APP, flags)
    result = run_capped_search_sync_group(
        TARGET, block, Budget(nonvolatile_cap=CAP, volatile_cap=CAP, accel_area_caps={}),
        warm_start_pf={}, shared_caches={}, verbose=False,
    )
    if result.array_counts.total_array_count > CAP:
        raise RuntimeError(f"greedy used {result.array_counts.total_array_count} arrays, over the cap of {CAP}")
    mapping = export_mapping(block, result)
    out = {"target": TARGET.name, "model": APP, "method": "greedy", "cap": CAP, "flags": flags,
           "array": {"rows": TARGET.geometry.rmu.rows, "cols": TARGET.geometry.rmu.cols},
           "mapping": mapping, "placement": place(mapping)}
    os.makedirs(RESULTS_DIR, exist_ok=True)
    with open(OUT_PATH, "w") as f:
        json.dump(out, f, indent=1, allow_nan=False)
    a = mapping["arrays"]
    print(f"{APP} at cap {CAP}: {a['total']} arrays ({a['nonvolatile']} stationary, peak {a['volatile']} working), "
          f"latency {mapping['latency_ns'] / 1e6:.2f} ms -> {OUT_PATH}")


if __name__ == "__main__":
    main()
