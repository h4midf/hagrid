"""Writes a converged (TimingOnlyMapping, HardwareMapping) state into the block as
mapped.schedule / mapped.stationary_placement ops (dialects/mapped_dialect.py). It
changes the IR, so it runs once on the settled state, never inside the search.

Three passes:
1. Before any IR change, the stationary-placement edges, the volatile ops' array ids,
   and every value's residency and stage: HardwareMapping finds consumers through
   .uses, which pass 2 redirects.
2. Every value of mapper.node_order gets a mapped.schedule after its op (a block
   argument's at the start of the block), and every other use moves to the wrapper's
   result (replace_uses_with_if, skipping the wrapper's own use of its operand).
3. The stationary placements, on the wrapped values.
"""

from __future__ import annotations

import heapq
from typing import Callable

from xdsl.ir import Block, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.mapped_dialect import MappedScheduleOp, MappedStationaryPlacementOp
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS, HardwareMapping, OperationMapping
from compiler.xdsl_timing import TimingOnlyMapping, _defining_op, _is_return


def _residency_of(value: SSAValue) -> str:
    """"nonvolatile" (a cim.save_stationary result), "volatile" (a MAPPED_OPERATION_OPS
    result: an op that occupies arrays) or "passthrough" (everything else, no arrays of
    its own)."""
    op = _defining_op(value)
    if isinstance(op, cd.CimSaveStationaryOp):
        return "nonvolatile"
    if isinstance(op, MAPPED_OPERATION_OPS):
        return "volatile"
    return "passthrough"


def _assign_volatile_array_ids(
    mapper: TimingOnlyMapping, hardware: HardwareMapping, operation_mappings: dict[SSAValue, OperationMapping]
) -> tuple[dict[SSAValue, list[int]], dict[SSAValue, list[int]], dict[str, int]]:
    """Array ids from the same demands as HardwareMapping.array_counts, each over a closed
    range (at equal times, starts before ends): every mapped op's volatile arrays
    (startup to output_end_time), and a held dynamic rhs's copy (rhs_hold_arrays,
    rhs_hold_start to rhs_hold_end), both in the op's pool (HardwareMapping.
    volatile_pool). A greedy interval coloring per pool: each opening takes the smallest
    free ids; opened in time order, it needs exactly the pool's peak number of ids, which
    materialize_schedule checks against array_counts. A pool's ids then follow the
    previous pools' arrays (volatile_pools), so every id is one array of the chip.
    Returns ({op: its arrays' ids}, {op: its held rhs's ids}, {pool: ids it used}).
    """
    events = []
    for value, mapping in operation_mappings.items():
        # mapper.node_index breaks equal-time ties deterministically (the peak does not
        # depend on the order)
        tie = mapper.node_index[value]
        pool = hardware.volatile_pool(value)
        for part, count, start, end in (
                ("op", int(mapping.volatile_arrays), mapping.startup, mapping.output_end_time),
                ("held", int(mapping.rhs_hold_arrays), mapping.rhs_hold_start, mapping.rhs_hold_end)):
            if count > 0:
                events.append((float(start), 0, tie, (pool, part, value), count))
                events.append((float(end), 1, tie, (pool, part, value), count))
    events.sort(key=lambda e: (e[0], e[1], e[2]))

    free_ids: dict[str, list[int]] = {}
    next_id: dict[str, int] = {}
    taken: dict[tuple, list[int]] = {}
    for _time, kind, _tie, key, count in events:
        pool = key[0]
        free = free_ids.setdefault(pool, [])
        if kind == 1:
            for i in taken[key]:
                heapq.heappush(free, i)
        else:
            ids = []
            for _ in range(count):
                if free:
                    ids.append(heapq.heappop(free))
                else:
                    ids.append(next_id.get(pool, 0))
                    next_id[pool] = ids[-1] + 1
            taken[key] = sorted(ids)
    split = {"op": {}, "held": {}}
    for (pool, part, value), ids in taken.items():
        split[part][value] = ids
    return split["op"], split["held"], next_id


def _assign_stages(node_order: list[SSAValue], specs: dict) -> dict[SSAValue, int]:
    """Stage of every value: the connected components of the [start_time, end_time)
    overlap graph, by merging overlapping intervals in (start_time, position) order. A
    zero-length window (stationary values, block arguments, const.ref) overlaps nothing
    and is its own stage."""
    order = sorted(range(len(node_order)), key=lambda i: (float(specs[i].start), i))
    stage_of: dict[SSAValue, int] = {}
    cur_stage = -1
    cur_max_end = None
    for i in order:
        value = node_order[i]
        spec = specs[i]
        start, end = float(spec.start), float(spec.end)
        if cur_max_end is None or start >= cur_max_end:
            cur_stage += 1
            cur_max_end = end
        else:
            cur_max_end = max(cur_max_end, end)
        stage_of[value] = cur_stage
    return stage_of


def materialize_schedule(
    block: Block,
    mapper: TimingOnlyMapping,
    hardware: HardwareMapping,
    sync_name_of: Callable[[SSAValue | None], str | None] | None = None,
) -> None:
    return_op = next(op for op in block.ops if _is_return(op))

    # -- pass 1: precompute everything against the UNMODIFIED IR --
    next_array_id = 0
    placement_specs: list[tuple[SSAValue, SSAValue, list[int], float, float]] = []
    for op in list(block.ops):
        if not isinstance(op, cd.CimSaveStationaryOp):
            continue
        stat_value = op.results[0]
        for consumer_op in hardware.stationary_consumers(stat_value):
            arrays_per_copy = hardware.stationary_arrays_per_copy(stat_value)
            copies = hardware.stationary_copies_for_consumer(consumer_op)
            total_arrays = int(arrays_per_copy) * int(copies)
            array_ids = list(range(next_array_id, next_array_id + total_arrays))
            next_array_id += total_arrays
            live_start = hardware.stationary_live_start(stat_value)
            live_end = hardware.stationary_live_end(consumer_op)
            placement_specs.append((stat_value, consumer_op.results[0], array_ids, live_start, live_end))

    operation_mappings = hardware.rebuild_operations()
    # the held rhs copies' ids only count toward the peak: the mapped IR has no op for them
    volatile_array_ids, _held_array_ids, used = _assign_volatile_array_ids(mapper, hardware, operation_mappings)
    counts = hardware.array_counts(hardware.rebuild(), operation_mappings)
    offset = {}
    base = 0
    for pool, peak, total in hardware.volatile_pools(counts):
        if used.get(pool, 0) != peak:
            raise RuntimeError(f"materialize: {used.get(pool, 0)} volatile array ids in {pool}, "
                               f"HardwareMapping counts {peak}")
        offset[pool] = base
        base += total
    volatile_array_ids = {v: [i + offset[hardware.volatile_pool(v)] for i in ids]
                          for v, ids in volatile_array_ids.items()}
    stage_of = _assign_stages(mapper.node_order, mapper.specs)

    # -- pass 2: wrap every value with its own mapped.schedule --
    wrapped_of: dict[SSAValue, SSAValue] = {}
    for idx, value in enumerate(mapper.node_order):
        spec = mapper.specs[idx]
        pf = int(mapper.mapped_pf[idx])
        sync_name = sync_name_of(mapper.sync_name.get(value)) if sync_name_of is not None and hasattr(mapper, "sync_name") else None
        residency = _residency_of(value)
        array_ids_for_value = volatile_array_ids.get(value) if residency == "volatile" else None

        wrapper = MappedScheduleOp(
            value,
            pf=pf,
            start_time=float(spec.start),
            end_time=float(spec.end),
            interval=float(spec.interval),
            unit_size=float(spec.unit_size),
            count=int(spec.count),
            result_type=value.type,
            sync_name=sync_name,
            residency=residency,
            array_ids=array_ids_for_value,
            stage=stage_of[value],
        )
        defining_op = _defining_op(value)
        if defining_op is None:
            first_op = block.first_op
            if first_op is None:
                block.add_op(wrapper)
            else:
                block.insert_op_before(wrapper, first_op)
        else:
            block.insert_op_after(wrapper, defining_op)
        value.replace_uses_with_if(wrapper.results[0], lambda use, w=wrapper: use.operation is not w)
        wrapped_of[value] = wrapper.results[0]

    # -- pass 3: emit stationary placements against the now-wrapped values --
    for stat_value, consumer_value, array_ids, live_start, live_end in placement_specs:
        placement = MappedStationaryPlacementOp(
            wrapped_of[stat_value], wrapped_of[consumer_value], array_ids,
            live_start=float(live_start), live_end=float(live_end),
        )
        block.insert_op_before(placement, return_op)
