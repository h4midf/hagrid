"""List scheduling for the RRAM/SRAM ensemble: list_schedule_rse.

The same algorithm as compiler/xdsl_list_schedule_1f1c.py (one forward pass in priority
order, choosing each array op's pf once), on the ensemble's own timing
(EnsembleTimingMapping) and macro counts (EnsembleHardwareMapping), within its budget: the
RRAM and SRAM macros together under one cap (EnsembleBudget). The target must be bound to
the app (targets.target_loader.ensemble_for_app), for its SRAM correction share. It
returns the same SearchResult as the greedy search (xdsl_search), so its mappings go
through the same export, tiles, area and energy passes.

  1. Start   every op at pf 1, the CMOS ops (xdsl_search._NONCRITICAL_MAXOUT_OPS) maxed
             out, as the greedy search starts. If that does not fit the budget, nothing
             smaller does: raise.
  2. Priority  each op's critical path to the sink: its own busy time (first output to
             last) plus its slowest consumer's priority, with every array op at its pf
             cap.
  3. Order   ops in a topological order that always takes the ready op of highest
             priority (Kahn's algorithm with a max-heap). An op is ready once every op
             producing one of its operands has been visited.
  4. Mode    for an array op, each pf of the ladder 1, 2, 4, ... up to its cap: its end
             from its producers' fixed timing, and whether the whole mapping fits the
             budget, with the visited ops at their chosen pf and the unvisited ones at pf 1
             (so every later op keeps its minimum). Of the pfs that fit, the earliest end,
             then the fewest macros. An array op's pf is, by its kind:
               matvec, stored rhs     RRAM copies, and with them its stationary SRAM
                                      correction macros (correction_plan: up to
                                      sram_macros_per_rram_macro per RRAM macro)
               matvec, both dynamic   SRAM PE copies of the rhs
               lut_search(_vector)    RRAM copies
  5. CMOS    after the pass, the CMOS ops are lowered as the greedy search lowers them
             (xdsl_search._lower_noncritical_pf_to_neighbor_rate).

Liveness is the macro model's: RRAM copies and SRAM correction macros are held for the
whole run; a dynamic matvec's SRAM macros from its start to its end, the peak over time
(EnsembleHardwareMapping).
"""
from __future__ import annotations

import heapq

from xdsl.ir import BlockArgument

from compiler.xdsl_accel_concurrency import accel_area_index
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_search import (
    EnsembleBudget, SearchMemo, SearchResult, _fits, _is_return, _lower_noncritical_pf_to_neighbor_rate,
    _maxout_noncritical_ops, _require_start_fits, _result, initial_mapping, rebuild_hardware, trial_array_counts,
)
from compiler.xdsl_timing import _defining_op
from targets.target_loader import EnsembleAppTarget


def _ladder(cap: int) -> list[int]:
    """1, 2, 4, ... up to cap (cap included)."""
    pfs, pf = [1], 1
    while pf < cap:
        pf = min(cap, pf * 2)
        pfs.append(pf)
    return pfs


def list_schedule_rse(target: EnsembleAppTarget, block, budget: EnsembleBudget) -> SearchResult:
    """The list schedule of `block` on the ensemble within `budget` (see the module docstring)."""
    if not isinstance(target, EnsembleAppTarget):
        raise TypeError(f"list_schedule_rse: {target.name} is not an ensemble target bound to an app "
                        "(targets.target_loader.ensemble_for_app)")
    mapper, hardware = initial_mapping(target, block)
    memo = SearchMemo()
    accel_index = accel_area_index(block, mapper.node_index)
    sink = next(op for op in block.ops if _is_return(op)).operands[0]
    order = mapper.node_order
    n = len(order)

    # 1. start: every op at pf 1, the CMOS ops maxed out
    _maxout_noncritical_ops(mapper, budget.accel_area_caps)
    _require_start_fits(mapper, rebuild_hardware(hardware), budget, accel_index)
    array_ops = [i for i, v in enumerate(order)
                 if not isinstance(v, BlockArgument) and type(_defining_op(v)) in MAPPED_OPERATION_OPS]

    # dependencies: every operand produced by another node (cat's included)
    producers: list[set] = [set() for _ in range(n)]
    consumers: list[list] = [[] for _ in range(n)]
    for i, v in enumerate(order):
        if isinstance(v, BlockArgument):
            continue
        for operand in _defining_op(v).operands:
            p = mapper.node_index.get(operand)
            if p is not None and p != i and p not in producers[i]:
                producers[i].add(p)
                consumers[p].append(i)

    # 2. priority: critical path to the sink, the array ops at their pf cap
    for i in array_ops:
        mapper.pf_overrides[i] = mapper.pf_cap_for_value(order[i])
    mapper.compute_all()
    prio = [0.0] * n
    for i in reversed(range(n)):
        spec = mapper.specs[i]
        busy = spec.end - spec.start if spec is not None else 0.0
        prio[i] = busy + max((prio[c] for c in consumers[i]), default=0.0)
    for i in array_ops:
        mapper.pf_overrides[i] = 1
    mapper.compute_all()

    # 3./4. one pass in priority order
    array_set = set(array_ops)
    unvisited_producers = [len(producers[i]) for i in range(n)]
    ready = [(-prio[i], i) for i in range(n) if unvisited_producers[i] == 0]
    heapq.heapify(ready)
    visited = 0
    while ready:
        _, i = heapq.heappop(ready)
        visited += 1
        if i in array_set:
            value = order[i]
            best = None
            for pf in _ladder(mapper.pf_cap_for_value(value)):
                if not _fits(mapper, hardware, memo, {value: pf}, budget, accel_index):
                    continue
                end = mapper.spec_at_pf(i, pf).end
                key = (end, trial_array_counts(mapper, hardware, memo, {value: pf}).total_array_count)
                if best is None or key < best[0]:
                    best = (key, pf)
            if best is None:
                # pf 1 fit at the start and every later op is still at pf 1
                raise RuntimeError(f"{_defining_op(value).name}: no pf fits, not even pf 1")
            if best[1] != 1:
                mapper.pf_overrides[i] = best[1]
                mapper.compute_from([value])
        for c in consumers[i]:
            unvisited_producers[c] -= 1
            if unvisited_producers[c] == 0:
                heapq.heappush(ready, (-prio[c], c))
    if visited != n:
        raise RuntimeError(f"list schedule visited {visited} of {n} nodes: the dependencies have a cycle")

    # 5. the CMOS ops, lowered as the greedy search lowers them
    _lower_noncritical_pf_to_neighbor_rate(mapper, hardware, memo, sink, budget, accel_index)
    mapper.compute_all()
    return _result(mapper, hardware, sink, rebuild_hardware(hardware))
