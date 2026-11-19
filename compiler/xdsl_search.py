"""The capped pf search: picks a parallelism factor (pf) per op
(run_capped_search_sync_group).

Each round collects candidate pf changes: forward walks from the graph's sources,
the critical path back from the sink, and sync groups (the mapped ops grouped by
their nearest sync point, offered whole and as every cost-ranked prefix, which finds
joint escalations a single-op walk cannot, e.g. Q/K/V together). It keeps those that
shorten the sink's latency and fit the budget, scores them by latency gain against
array cost, and applies the best disjoint batch. Timing comes from TimingOnlyMapping,
array counts from HardwareMapping; no IR is created or changed.

Sync points (annotate_syncs) split the graph into groups between merge points; a
local pf change only escalates parents in the same group.
"""

from __future__ import annotations

import hashlib
import time
from collections import defaultdict, deque
from dataclasses import dataclass

from xdsl.ir import BlockArgument, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.xdsl_accel_concurrency import (
    ACCEL_OP_KINDS, accel_area_index, accel_kind_peak_area_um2, accel_kinds_peak_area_um2,
)
from compiler.xdsl_dual_module_hardware import DualModuleHardwareMapping
from compiler.xdsl_dual_module_timing import DualModuleTimingMapping
from compiler.xdsl_ensemble_hardware import EnsembleHardwareMapping
from compiler.xdsl_ensemble_timing import EnsembleTimingMapping
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS, HardwareArrayCounts, HardwareMapping, StationaryPlacement
from compiler.xdsl_timing import SHAPE_OPS, TimingOnlyMapping, _defining_op, _is_return
from targets.target_loader import DualModuleTarget, EnsembleAppTarget, EnsembleTarget, SingleModuleTarget

LATENCY_WEIGHT = 1.0
ARRAY_WEIGHT = 5.0

# ops with no array cost and nothing to trade off: their pf is maxed before the round
# loop (_maxout_noncritical_ops) instead of searched
_NONCRITICAL_MAXOUT_OPS = (
    cd.CimReduceVecOp, cd.CimVecElMulOp, cd.CimVecAddOp, cd.CimArgmaxOp, cd.CimMaxOp,
    cd.CimReluOp, cd.CimGeluOp, cd.CimSoftmaxOp, cd.CimSiluOp, cd.CimEluOp, cd.CimLeakyReluOp,
    cd.CimQuantizeOp, cd.CimDequantizeOp, cd.CimLayerNormOp, cd.CimRMSNormOp,
    cd.CimRowAbsmaxOp, cd.CimQuantizePerTokenOp, cd.CimVecSelectOp, cd.CimRsqrtOp,
)
# set for the hot `type(op) in ...` check
_NONCRITICAL_MAXOUT_OPS_SET = frozenset(_NONCRITICAL_MAXOUT_OPS)

_CRITICAL_SIBLING_TOL = 1e-6
# how much later lowering one non-critical op may make the sink finish, as a share of the
# latency at that point (our choice: a few ns of latency must not keep hundreds of mm^2
# of circuits). Each op is checked on its own, so the total can grow by more
_LOWERING_LATENCY_STEP = 0.005
# and in total, as a share of the search's latency before any lowering
_LOWERING_LATENCY_TOTAL = 0.15


# ---------------------------------------------------------------------------
# graph topology


def _dynamic_ancestors(mapper: TimingOnlyMapping, value: SSAValue, memo: dict) -> set:
    cached = memo.get(value)
    if cached is not None:
        return cached
    ancestors: set = set()
    for parent in mapper.dynamic_parents(value):
        ancestors.add(parent)
        ancestors.update(_dynamic_ancestors(mapper, parent, memo))
    memo[value] = ancestors
    return ancestors


def _all_operands(op) -> list[SSAValue]:
    """All operands of `op`: unlike TimingOnlyMapping.dynamic_parents, block arguments
    and save_stationary included. cat has none, as in dynamic_parents."""
    if type(op) is cd.CimCatOp:
        return []
    return list(op.operands)


def _forward_consumers(mapper: TimingOnlyMapping, value: SSAValue) -> list[SSAValue]:
    """Consumers of `value` in creation order. SSAValue.uses lists them newest first,
    and _forward_walk treats the first consumer as the main branch, so they are
    sorted by node position.
    """
    seen = {}
    for use in value.uses:
        op = use.operation
        if _is_return(op) or not op.results:
            continue
        seen[id(op)] = op.results[0]
    return sorted(seen.values(), key=lambda v: mapper.node_index[v])


# ---------------------------------------------------------------------------
# annotate_syncs


@dataclass(frozen=True)
class SyncInfo:
    """Sync points of a block (annotate_syncs), per node."""
    # nearest sync point at or downstream of the node (_resolve_local groups by it)
    nearest_sync: dict[SSAValue, SSAValue | None]
    is_sync: dict[SSAValue, bool]
    # for a sync point, the next one downstream (_build_live_groups)
    next_sync: dict[SSAValue, SSAValue | None]


def annotate_syncs(mapper: TimingOnlyMapping) -> SyncInfo:
    """Sync points of the mapper's block: a merge whose parents sit at different
    depths, plus the sink. Uses only the node order (IR structure), not the timing."""
    node_order = mapper.node_order

    depth: dict[SSAValue, int] = {}
    for value in node_order:
        parents = mapper.dynamic_parents(value)
        depth[value] = 0 if not parents else 1 + max(depth[p] for p in parents)

    is_sync: dict[SSAValue, bool] = {}
    ancestor_memo: dict = {}
    for value in node_order:
        parents = mapper.dynamic_parents(value)
        if len(parents) < 2:
            continue
        low = min(depth[p] for p in parents)
        high = max(depth[p] for p in parents)
        if high <= low:
            continue

        is_sync[value] = True
        for anc in _dynamic_ancestors(mapper, value, ancestor_memo):
            if low < depth[anc] <= high:
                is_sync[anc] = False

    return_op = next(op for op in mapper.block.ops if _is_return(op))
    sink_value = return_op.operands[0]
    is_sync[sink_value] = True

    nearest_sync: dict[SSAValue, SSAValue] = {}
    queue: deque = deque()
    for value, flag in is_sync.items():
        if flag:
            nearest_sync[value] = value
            queue.append(value)
    while queue:
        cur = queue.popleft()
        cur_op = _defining_op(cur)
        if cur_op is None:
            continue
        # back over all operands, not just dynamic_parents
        for arg in _all_operands(cur_op):
            if arg in nearest_sync:
                continue
            nearest_sync[arg] = nearest_sync[cur]
            queue.append(arg)

    # next_sync: forward BFS from each sync point to the first other sync point
    sync_values = [value for value, flag in is_sync.items() if flag]
    next_sync: dict[SSAValue, SSAValue] = {}
    for sync in sync_values:
        seen = {sync}
        sync_queue: deque = deque(
            c for c in _forward_consumers(mapper, sync) if c not in seen
        )
        seen.update(sync_queue)
        found = None
        while sync_queue:
            cur = sync_queue.popleft()
            if is_sync.get(cur, False):
                found = cur
                break
            for consumer in _forward_consumers(mapper, cur):
                if consumer not in seen:
                    seen.add(consumer)
                    sync_queue.append(consumer)
        next_sync[sync] = found

    return SyncInfo(
        nearest_sync={value: nearest_sync.get(value) for value in node_order},
        is_sync={value: is_sync.get(value, False) for value in node_order},
        next_sync={value: next_sync.get(value) for value in node_order},
    )


# ---------------------------------------------------------------------------
# local/backward pf resolution


def _next_pf(pf: int, pf_cap: int) -> int:
    """The next pf the search tries: double, capped."""
    return min(pf_cap, pf * 2)


def _is_constraining(mapper: TimingOnlyMapping, value: SSAValue, parent: SSAValue, changes: dict) -> bool:
    current_pf = mapper.mapped_pf[mapper.node_index[parent]]
    pf_cap = mapper.pf_cap_for_value(parent)
    if current_pf >= pf_cap:
        return False
    trial_pf = _next_pf(current_pf, pf_cap)
    without = mapper.propagate_and_measure(changes, value) if changes else 0.0
    with_parent = dict(changes)
    with_parent[parent] = trial_pf
    with_ = mapper.propagate_and_measure(with_parent, value)
    return with_ > without


def _resolve_local(mapper: TimingOnlyMapping, sync: SyncInfo, value: SSAValue, changes: dict, visited: set) -> None:
    if value in visited:
        return
    visited.add(value)
    if value in changes:
        return
    value_sync = sync.nearest_sync[value]
    for parent in mapper.dynamic_parents(value):
        if parent in visited:
            continue
        if sync.nearest_sync[parent] is not value_sync:
            continue
        if not _is_constraining(mapper, value, parent, changes):
            continue
        _resolve_local(mapper, sync, parent, changes, visited)
    pf_cap = mapper.pf_cap_for_value(value)
    current_pf = mapper.mapped_pf[mapper.node_index[value]]
    if current_pf >= pf_cap:
        return
    changes[value] = _next_pf(current_pf, pf_cap)


def _resolve_backward(mapper: TimingOnlyMapping, value: SSAValue, changes: dict) -> None:
    # the visited set holds node positions, not SSAValues (_resolve_backward_at)
    _resolve_backward_at(mapper, mapper.node_index[value], changes, set())


def _resolve_backward_at(mapper: TimingOnlyMapping, idx: int, changes: dict, visited: set) -> None:
    """_resolve_backward's recursion, by node position: int lookups instead of
    SSAValue hashes, since it runs millions of times per search. `changes` stays
    SSAValue-keyed."""
    if idx in visited:
        return
    visited.add(idx)
    value = mapper.node_order[idx]
    if value in changes:
        return
    for parent_idx in mapper.dynamic_parent_indices(idx):
        _resolve_backward_at(mapper, parent_idx, changes, visited)
    pf_cap = mapper.pf_cap_at(idx)
    current_pf = mapper.mapped_pf[idx]
    if current_pf >= pf_cap:
        return
    changes[value] = _next_pf(current_pf, pf_cap)


def _bump_joint_until_helps(mapper: TimingOnlyMapping, changes: dict, values: list, measured_at: SSAValue,
                            baseline: float) -> bool:
    pf_caps = {v: mapper.pf_cap_for_value(v) for v in values}
    trial = dict(changes)
    for v in values:
        trial.setdefault(v, mapper.mapped_pf[mapper.node_index[v]])

    while any(trial[v] < pf_caps[v] for v in values):
        for v in values:
            trial[v] = _next_pf(trial[v], pf_caps[v])
        if mapper.propagate_and_measure(trial, measured_at) > baseline:
            changes.update(trial)
            return True
    return False


# ---------------------------------------------------------------------------
# find_pf_candidates / _forward_walk


def find_pf_candidates(mapper: TimingOnlyMapping, sync: SyncInfo) -> list[dict]:
    visited: set = set()
    # sources: op results (not block arguments or constants) with no dynamic parents
    frontier = [
        value for value in mapper.node_order
        if type(value) is not BlockArgument
        and type(value.owner) is not ConstRefOp
        and len(mapper.dynamic_parents(value)) == 0
    ]
    candidates = []
    extra_candidates = []

    while frontier:
        value = frontier.pop()
        if value in visited:
            continue
        result = _forward_walk(mapper, sync, value, visited, frontier, extra_candidates)
        if result:
            candidates.append(result)

    candidates.extend(extra_candidates)
    return candidates


def _forward_walk(mapper: TimingOnlyMapping, sync: SyncInfo, start_value: SSAValue, visited: set, frontier: list,
                  extra_candidates: list) -> dict:
    visited.add(start_value)
    changes: dict = {}
    changes_overlay: dict = {}
    changes_base: dict = {}
    best: dict = {}
    current = start_value

    while True:
        consumers = _forward_consumers(mapper, current)
        if not consumers:
            if changes:
                benefit, _ = mapper.propagate_and_measure_chained(changes, current, changes_overlay, changes_base)
                if benefit > 0:
                    best = changes
            return best

        consumer, extra_consumers = consumers[0], consumers[1:]
        for extra in extra_consumers:
            if extra not in visited:
                frontier.append(extra)

        if consumer in visited:
            current = consumer
            continue

        if changes:
            benefit_a, _ = mapper.propagate_and_measure_chained(changes, consumer, changes_overlay, changes_base)
            if benefit_a > 0:
                best = changes
                changes = {}
                changes_overlay = {}
                changes_base = {}

        next_hop = _forward_consumers(mapper, consumer)
        measured_at = next_hop[0] if next_hop else consumer

        local_changes: dict = {}
        _resolve_local(mapper, sync, consumer, local_changes, set())
        if local_changes and mapper.propagate_and_measure(local_changes, measured_at) > 0:
            extra_candidates.append(dict(local_changes))

        _resolve_backward(mapper, consumer, changes)

        if changes:
            current_benefit, trial_overlay = mapper.propagate_and_measure_chained(
                changes, measured_at, changes_overlay, changes_base
            )
        else:
            current_benefit, trial_overlay = 0.0, dict(changes_overlay)
        helped = bool(changes) and current_benefit > 0
        if helped:
            changes_overlay = trial_overlay
            changes_base = dict(changes)
        if not helped:
            if _bump_joint_until_helps(
                mapper, changes, list(changes.keys()) + [consumer], measured_at, baseline=current_benefit
            ):
                helped = True
                changes_overlay = {}
                changes_base = {}

        if not helped:
            for c in _forward_consumers(mapper, consumer):
                if c not in visited:
                    frontier.append(c)
            return best

        best = changes
        visited.add(consumer)
        current = consumer


# ---------------------------------------------------------------------------
# find_critical_path_candidates


def _critical_siblings(mapper: TimingOnlyMapping, value: SSAValue) -> list[SSAValue]:
    parents = mapper.dynamic_parents(value)
    if len(parents) <= 1:
        return parents
    starts = {p: mapper.specs[mapper.node_index[p]].start for p in parents}
    intervals = {p: mapper.specs[mapper.node_index[p]].interval for p in parents}
    ends = {p: mapper.specs[mapper.node_index[p]].end for p in parents}
    max_start = max(starts.values())
    max_interval = max(intervals.values())
    max_end = max(ends.values())
    return [
        p for p in parents
        if starts[p] >= max_start - _CRITICAL_SIBLING_TOL
        or intervals[p] >= max_interval - _CRITICAL_SIBLING_TOL
        or ends[p] >= max_end - _CRITICAL_SIBLING_TOL
    ]


def _backward_critical_set(mapper: TimingOnlyMapping, sink: SSAValue) -> list[SSAValue]:
    visited: set = set()
    frontier = [sink]
    order = []
    while frontier:
        value = frontier.pop()
        if value in visited:
            continue
        visited.add(value)
        order.append(value)
        for parent in _critical_siblings(mapper, value):
            if parent not in visited:
                frontier.append(parent)
    return order


def _critical_path_standalone_candidates(mapper: TimingOnlyMapping, sink: SSAValue, chain: list) -> list[dict]:
    candidates = []
    for value in chain:
        if mapper.mapped_pf[mapper.node_index[value]] >= mapper.pf_cap_for_value(value):
            continue
        changes: dict = {}
        _resolve_backward(mapper, value, changes)
        if changes and mapper.propagate_and_measure(changes, sink) > 0:
            candidates.append(changes)
    return candidates


def _critical_path_greedy_escalation(mapper: TimingOnlyMapping, sink: SSAValue, chain: list) -> list[dict]:
    _GREEDY_MAX_PASSES = 5
    greedy_changes: dict = {}
    greedy_overlay: dict = {}
    greedy_benefit = 0.0
    for _ in range(_GREEDY_MAX_PASSES):
        changed_this_pass = False
        for value in chain:
            current_pf = greedy_changes.get(value, mapper.mapped_pf[mapper.node_index[value]])
            pf_cap = mapper.pf_cap_for_value(value)
            if current_pf >= pf_cap:
                continue
            trial_pf = _next_pf(current_pf, pf_cap)
            trial = dict(greedy_changes)
            trial[value] = trial_pf
            benefit, trial_overlay = mapper.propagate_and_measure_chained(trial, sink, greedy_overlay, greedy_changes)
            if benefit > greedy_benefit:
                greedy_changes[value] = trial_pf
                greedy_overlay = trial_overlay
                greedy_benefit = benefit
                changed_this_pass = True
        if not changed_this_pass:
            break
    return [greedy_changes] if greedy_changes else []


def find_critical_path_candidates(mapper: TimingOnlyMapping, sink: SSAValue) -> list[dict]:
    chain = _backward_critical_set(mapper, sink)
    candidates = _critical_path_standalone_candidates(mapper, sink, chain)
    candidates.extend(_critical_path_greedy_escalation(mapper, sink, chain))
    return candidates


# ---------------------------------------------------------------------------
# scoring / hardware rebuild / round loop


def rebuild_hardware(hardware: HardwareMapping) -> HardwareArrayCounts:
    """Array totals at the mapper's current pf (the hardware reads the mapper it was built with)."""
    return hardware.array_counts(hardware.rebuild(), hardware.rebuild_operations())


def initial_mapping(target: SingleModuleTarget | DualModuleTarget | EnsembleAppTarget, block):
    """Mapper and hardware of `block` on `target` at pf 1 everywhere (per target type)."""
    match target:
        case SingleModuleTarget():
            mapper = TimingOnlyMapping(target, block)
            mapper.compute_all()
            hardware = HardwareMapping(target, block, mapper)
        case DualModuleTarget():
            mapper = DualModuleTimingMapping(target, block)
            mapper.compute_all()
            hardware = DualModuleHardwareMapping(target, block, mapper)
        case EnsembleAppTarget():
            mapper = EnsembleTimingMapping(target, block)
            mapper.compute_all()
            hardware = EnsembleHardwareMapping(target, block, mapper)
        case EnsembleTarget():
            raise TypeError(f"{target.name}: bind the target to an app first (targets.target_loader.ensemble_for_app)")
        case _:
            raise TypeError(f"no mapper for target type {type(target).__name__}")
    return mapper, hardware


def _maxout_noncritical_ops(mapper: TimingOnlyMapping, accel_area_caps: dict[str, float]) -> None:
    """Sets every _NONCRITICAL_MAXOUT_OPS op to its pf cap before the round loop. The
    round loop only raises a pf that shortens the sink's latency, so an op off the
    critical path would otherwise stay at pf 1. A large pf keeps pipelining: each
    circuit starts a problem as soon as its inputs arrive (xdsl_timing._pipelined_spec).

    Accelerator kinds named in accel_area_caps ({kind: um^2}, each its own budget; {} =
    none) are not maxed outright: _budget_aware_maxout_kind picks the largest pf that
    fits the kind's budget."""
    accel_kinds = {op_type for op_type, kind in ACCEL_OP_KINDS.items() if kind in accel_area_caps}
    for i, value in enumerate(mapper.node_order):
        op = _defining_op(value)
        if type(op) in _NONCRITICAL_MAXOUT_OPS_SET and type(op) not in accel_kinds:
            mapper.pf_overrides[i] = mapper.pf_cap_for_value(value)
    mapper.compute_all()

    kind_to_op_type = {kind: op_type for op_type, kind in ACCEL_OP_KINDS.items()}
    for kind, cap_um2 in accel_area_caps.items():
        _budget_aware_maxout_kind(mapper, kind_to_op_type[kind], kind, cap_um2)


def _budget_aware_maxout_kind(mapper: TimingOnlyMapping, op_type: type, kind: str, area_cap_um2: float) -> None:
    """Raises every op of one accelerator kind to the same fraction of its pf cap,
    binary-searching the largest fraction whose peak area (accel_kind_peak_area_um2)
    fits area_cap_um2. If even pf 1 does not fit, the ops stay at pf 1."""
    entries: list[tuple[int, int]] = []
    for value in mapper.node_order:
        op = _defining_op(value)
        if type(op) is not op_type:
            continue
        cap = mapper.pf_cap_for_value(value)
        if cap <= 1:
            continue
        entries.append((mapper.node_index[value], cap))
    if not entries:
        return

    baseline_overrides = dict(mapper.pf_overrides)
    lo, hi = 0.0, 1.0
    best_overrides = None
    for _ in range(20):  # 2^-20 resolution on the fraction
        mid = (lo + hi) / 2.0
        trial = dict(baseline_overrides)
        for idx, cap in entries:
            trial[idx] = max(1, min(cap, int(round(cap * mid))))
        mapper.pf_overrides = trial
        mapper.compute_all()
        area = accel_kind_peak_area_um2(mapper.block, mapper.specs, mapper.mapped_pf, mapper.node_index, kind)
        if area <= area_cap_um2:
            best_overrides = trial
            lo = mid
        else:
            hi = mid
    mapper.pf_overrides = best_overrides if best_overrides is not None else baseline_overrides
    mapper.compute_all()


class SearchMemo:
    """Results the search memoizes, keyed on the full pf state (mapper.memo_state, set
    by the search while specs is exactly compute_all of pf_overrides): array counts
    of trial changes, a round's candidate list, per-kind accelerator peak areas.
    """

    def __init__(self):
        self.array_counts: dict = {}
        self.candidates: dict = {}
        self.accel_area: dict = {}


def _score(mapper, hardware, memo: SearchMemo, changes, top_level_sink, baseline_latency, baseline_arrays) -> float:
    benefit = mapper.propagate_and_measure(changes, top_level_sink)
    array_delta = _array_cost_delta(mapper, hardware, memo, changes, baseline_arrays)
    latency_term = benefit / baseline_latency
    array_term = array_delta / baseline_arrays
    return LATENCY_WEIGHT * latency_term - ARRAY_WEIGHT * array_term


def with_trial(mapper, changes: dict, measure):
    """measure() while `changes` are applied to the mapper; the mapper is restored after.
    Only the nodes the change touches are saved and restored."""
    overlay = None
    if mapper.memo_state is not None:
        overlay = mapper.reusable_overlay(frozenset(changes.items()), changes)

    changed_idxs = [mapper.node_index[value] for value in changes]
    touched = set(overlay) if overlay is not None else set(mapper.affected_indices(changes.keys()))
    touched.update(changed_idxs)
    saved_specs = {i: mapper.specs[i] for i in touched}
    saved_mapped_pf = {i: mapper.mapped_pf[i] for i in touched}
    saved_overrides = {i: mapper.pf_overrides[i] for i in changed_idxs if i in mapper.pf_overrides}
    saved_memo_state = mapper.memo_state

    try:
        for value, pf in changes.items():
            mapper.pf_overrides[mapper.node_index[value]] = pf
        if overlay is not None:
            # the overlay propagate_and_measure already computed is exactly what
            # compute_from would write (reusable_overlay)
            mapper.specs.update(overlay)
            for value, pf in changes.items():
                mapper.mapped_pf[mapper.node_index[value]] = pf
        else:
            mapper.compute_from(changes.keys())
        return measure()
    finally:
        mapper.specs.update(saved_specs)
        mapper.mapped_pf.update(saved_mapped_pf)
        for i in changed_idxs:
            if i in saved_overrides:
                mapper.pf_overrides[i] = saved_overrides[i]
            else:
                del mapper.pf_overrides[i]
        mapper.memo_state = saved_memo_state


def trial_array_counts(mapper, hardware, memo: SearchMemo, changes: dict) -> HardwareArrayCounts:
    """Array counts if `changes` were applied. Memoized in memo.array_counts on
    (memo_state, changes) while the search has set memo_state."""
    memo_key = None
    if mapper.memo_state is not None:
        memo_key = (mapper.memo_state, frozenset(changes.items()))
        cached = memo.array_counts.get(memo_key)
        if cached is not None:
            return cached
    counts = with_trial(mapper, changes, lambda: rebuild_hardware(hardware))
    if memo_key is not None:
        memo.array_counts[memo_key] = counts
    return counts


def _array_cost_delta(mapper, hardware, memo: SearchMemo, changes, baseline_arrays) -> int:
    """Change in total arrays if `changes` were applied."""
    return trial_array_counts(mapper, hardware, memo, changes).total_array_count - baseline_arrays


# ---------------------------------------------------------------------------
# sync groups

_TRIVIAL_GROUP_SIZE = 2


def _build_live_groups(mapper: TimingOnlyMapping, sync: SyncInfo) -> dict:
    """Mapped ops grouped by their nearest sync point; a sync point's group absorbs
    the next sync point's group when that one is trivially small (one hop only).
    The search drops members from these lists as they reach their pf cap."""
    groups: dict = defaultdict(list)
    for value in mapper.node_order:
        if type(_defining_op(value)) in MAPPED_OPERATION_OPS:
            groups[sync.nearest_sync[value]].append(value)
    for value in mapper.node_order:
        if not sync.is_sync[value]:
            continue
        next_sync = sync.next_sync[value]
        if next_sync is None or value not in groups or next_sync not in groups:
            continue
        if len(groups[next_sync]) <= _TRIVIAL_GROUP_SIZE:
            groups[value].extend(groups.pop(next_sync))
    return dict(groups)


def _escalatable_groups(mapper: TimingOnlyMapping, live_groups: dict) -> list:
    """Per group, the members below their pf cap with their next pf. pf only
    increases, so a member at its cap is dropped from live_groups for good."""
    result = []
    for key in list(live_groups.keys()):
        escalatable = []
        still_live = []
        for value in live_groups[key]:
            pf_cap = mapper.pf_cap_for_value(value)
            current_pf = mapper.mapped_pf[mapper.node_index[value]]
            if current_pf < pf_cap:
                escalatable.append((value, _next_pf(current_pf, pf_cap)))
                still_live.append(value)
        if still_live:
            live_groups[key] = still_live
        else:
            del live_groups[key]
        if escalatable:
            result.append(escalatable)
    return result


def _build_prefix_candidates(escalatable_groups: list, deltas: dict) -> list:
    """Per group: the whole group, plus every prefix of its members sorted by array
    cost. deltas: {(group_idx, member_idx): array cost delta}."""
    candidates = []
    for group_idx, escalatable in enumerate(escalatable_groups):
        full_changes = {value: new_pf for value, new_pf in escalatable}
        candidates.append(full_changes)
        if len(escalatable) <= 1:
            continue
        costed = [
            (deltas[(group_idx, member_idx)], value, new_pf)
            for member_idx, (value, new_pf) in enumerate(escalatable)
        ]
        costed.sort(key=lambda t: t[0])
        prefix_changes: dict = {}
        for _delta, value, new_pf in costed[:-1]:  # last prefix == full_changes, already added
            prefix_changes = dict(prefix_changes)
            prefix_changes[value] = new_pf
            candidates.append(dict(prefix_changes))
    return candidates


def _find_sync_group_candidates(mapper: TimingOnlyMapping, hardware: HardwareMapping, memo: SearchMemo,
                                live_groups: dict, baseline_arrays: int) -> list:
    escalatable_groups = _escalatable_groups(mapper, live_groups)
    deltas = {}
    for group_idx, escalatable in enumerate(escalatable_groups):
        for member_idx, (value, new_pf) in enumerate(escalatable):
            deltas[(group_idx, member_idx)] = _array_cost_delta(
                mapper, hardware, memo, {value: new_pf}, baseline_arrays
            )
    return _build_prefix_candidates(escalatable_groups, deltas)


# ---------------------------------------------------------------------------
# budget, batching, the search


@dataclass(frozen=True)
class Budget:
    """What a pf change must fit: array caps per memory kind, and a peak-area cap per
    accelerator kind ({} caps none)."""
    nonvolatile_cap: int
    volatile_cap: int
    accel_area_caps: dict[str, float]


@dataclass(frozen=True)
class DualModuleBudget:
    """What a pf change must fit on a dual-module target: a cap on the total arrays of
    both modules (DualModuleArrayCounts.total_array_count), and a peak-area cap per
    accelerator kind ({} caps none)."""
    array_cap: int
    accel_area_caps: dict[str, float]


@dataclass(frozen=True)
class EnsembleBudget:
    """What a pf change must fit on the RRAM/SRAM ensemble: a cap on its RRAM + SRAM
    macros (EnsembleArrayCounts.total_array_count), and a peak-area cap per accelerator
    kind ({} caps none)."""
    macro_cap: int
    accel_area_caps: dict[str, float]


def _arrays_over_budget(counts, budget: Budget | DualModuleBudget | EnsembleBudget) -> str | None:
    """Why `counts` is over the array caps of `budget`, None if within them."""
    match budget:
        case Budget():
            if counts.nonvolatile_array_count > budget.nonvolatile_cap:
                return f"{counts.nonvolatile_array_count} nonvolatile arrays, over the cap of {budget.nonvolatile_cap}"
            if counts.volatile_array_count > budget.volatile_cap:
                return f"{counts.volatile_array_count} volatile arrays, over the cap of {budget.volatile_cap}"
        case DualModuleBudget():
            if counts.total_array_count > budget.array_cap:
                return f"{counts.total_array_count} arrays (analog + digital), over the cap of {budget.array_cap}"
        case EnsembleBudget():
            if counts.total_array_count > budget.macro_cap:
                return f"{counts.total_array_count} macros (RRAM + SRAM), over the cap of {budget.macro_cap}"
        case _:
            raise TypeError(f"unknown budget type {type(budget).__name__}")
    return None


@dataclass(frozen=True)
class SearchResult:
    latency_ns: float
    array_counts: HardwareArrayCounts
    mapper: TimingOnlyMapping
    hardware: HardwareMapping
    stationary_placements: list[StationaryPlacement]
    pf: dict[SSAValue, int]


def _busy_ns(spec) -> float:
    """How long an op takes over its whole output: first output batch to last."""
    return spec.end - spec.start


def _neighbor_busy_ns(mapper: TimingOnlyMapping, value: SSAValue) -> float:
    """The longest busy time (_busy_ns) among the op's dynamic producers and its
    consumers, in the current mapping: the rate the op has to keep up with. An op
    faster than both its neighbours gains nothing: its producer cannot feed it faster
    and its consumer cannot take its output faster. Shape ops (xdsl_timing.SHAPE_OPS)
    only pass their operand's timing on, so the neighbours are looked for through them."""
    neighbors = []
    producers = list(mapper.dynamic_parents(value))
    while producers:
        p = producers.pop()
        if type(_defining_op(p)) in SHAPE_OPS:
            producers.extend(mapper.dynamic_parents(p))
        else:
            neighbors.append(p)
    consumers = [use.operation for use in value.uses]
    while consumers:
        op = consumers.pop()
        if type(op) in SHAPE_OPS:
            consumers.extend(use.operation for result in op.results for use in result.uses)
            continue
        neighbors.extend(r for r in op.results if r in mapper.node_index)
    if not neighbors:
        raise RuntimeError(f"{_defining_op(value).name}: no producer or consumer to match its pf to")
    return max(_busy_ns(mapper.specs[mapper.node_index[n]]) for n in neighbors)


def _lower_noncritical_pf_to_neighbor_rate(mapper: TimingOnlyMapping, hardware: HardwareMapping,
                                           memo: SearchMemo, sink, budget: Budget, accel_index: tuple) -> None:
    """After the search: lowers the pf of each op _maxout_noncritical_ops raised to the
    smallest value (binary search) whose busy time is no longer than the slower of its
    neighbours' (_neighbor_busy_ns), or than its own at the current pf (an op fed at its
    producer's rate is busy a round longer than the producer whatever its pf), that keeps
    the sink within _LOWERING_LATENCY_STEP of the latency before this op is lowered and
    within _LOWERING_LATENCY_TOTAL of the search's latency, and that still fits the
    budget. More circuits than that are
    shared: the op would only wait for its producer or its consumer.
    Ops are lowered from the last to the first, so an op's consumers have already been
    lowered and it matches the rate they really run at; through a chain of these ops
    (a layer norm's vec_add, vec_el_mul and reduce_vec) the slowest rate reaches back to
    the chain's start. Its producers are still at their maxed pf. The sink check bounds
    the latency: matching busy times alone can still start a consumer later.
    Lowering moves when downstream arrays are live, so the array counts are checked too."""
    sink_idx = mapper.node_index[sink]
    total_limit = mapper.specs[sink_idx].end * (1 + _LOWERING_LATENCY_TOTAL)
    # the sink itself too: the sink check keeps it within the limit
    for i in reversed(range(sink_idx + 1)):
        value = mapper.node_order[i]
        if type(_defining_op(value)) not in _NONCRITICAL_MAXOUT_OPS_SET:
            continue
        current = mapper.mapped_pf[i]
        if current <= 1:
            continue
        bound_ns = max(_neighbor_busy_ns(mapper, value), _busy_ns(mapper.specs[i]))
        latency_limit = min(mapper.specs[sink_idx].end * (1 + _LOWERING_LATENCY_STEP), total_limit)

        def acceptable(pf: int) -> bool:
            if _busy_ns(mapper.spec_at_pf(i, pf)) > bound_ns:
                return False
            if mapper.specs[sink_idx].end - mapper.propagate_and_measure({value: pf}, sink) > latency_limit:
                return False
            return _fits(mapper, hardware, memo, {value: pf}, budget, accel_index)

        lo, hi = 1, current  # pf=hi is the current state: acceptable
        while lo < hi:
            mid = (lo + hi) // 2
            if acceptable(mid):
                hi = mid
            else:
                lo = mid + 1
        if hi < current:
            mapper.pf_overrides[i] = hi
            mapper.compute_from([value])


def _trial_accel_areas(mapper, memo: SearchMemo, changes: dict, accel_index: tuple,
                       accel_area_caps: dict[str, float]) -> dict[str, float]:
    """Peak area (um^2) of every capped accelerator kind if `changes` were applied.
    Memoized in memo.accel_area on (memo_state, changes)."""
    kinds = tuple(accel_area_caps)
    if not kinds:
        return {}
    memo_key = None
    if mapper.memo_state is not None:
        memo_key = (mapper.memo_state, frozenset(changes.items()), kinds)
        cached = memo.accel_area.get(memo_key)
        if cached is not None:
            return cached
    areas = with_trial(mapper, changes,
                       lambda: accel_kinds_peak_area_um2(accel_index, mapper.specs, mapper.mapped_pf, kinds))
    if memo_key is not None:
        memo.accel_area[memo_key] = areas
    return areas


def _fits(mapper, hardware: HardwareMapping, memo: SearchMemo, changes: dict, budget: Budget,
          accel_index: tuple) -> bool:
    """True if `changes` keeps both array counts and every capped accelerator kind's
    peak area within the budget."""
    counts = trial_array_counts(mapper, hardware, memo, changes)
    if _arrays_over_budget(counts, budget) is not None:
        return False
    areas = _trial_accel_areas(mapper, memo, changes, accel_index, budget.accel_area_caps)
    return all(areas[kind] <= cap for kind, cap in budget.accel_area_caps.items())


def _usable_candidates(candidate_sets, mapper, hardware: HardwareMapping, memo: SearchMemo, sink,
                       baseline_latency: float, baseline_arrays: int, budget: Budget, accel_index: tuple) -> list:
    """(changes, score) of every candidate that shortens the sink's latency and fits."""
    usable = []
    for changes in candidate_sets:
        if mapper.propagate_and_measure(changes, sink) <= 0:
            continue
        if not _fits(mapper, hardware, memo, changes, budget, accel_index):
            continue
        usable.append((changes, _score(mapper, hardware, memo, changes, sink, baseline_latency, baseline_arrays)))
    return usable


def _pick_batch(usable: list, mapper, hardware: HardwareMapping, memo: SearchMemo, sink, budget: Budget,
                accel_index: tuple):
    """Greedily combines disjoint usable candidates, best score first. Each addition
    is kept only if the combined change measures better and still fits."""
    ranked = sorted(usable, key=lambda pair: pair[1], reverse=True)
    batch = dict(ranked[0][0])
    included = [ranked[0][0]]
    used_values = set(batch.keys())
    best_benefit = mapper.propagate_and_measure(batch, sink)

    for changes, _score_value in ranked[1:]:
        candidate_values = set(changes.keys())
        if candidate_values & used_values:
            continue
        trial_batch = dict(batch)
        trial_batch.update(changes)
        trial_benefit = mapper.propagate_and_measure(trial_batch, sink)
        if trial_benefit <= best_benefit:
            continue
        if not _fits(mapper, hardware, memo, trial_batch, budget, accel_index):
            continue
        batch = trial_batch
        included.append(changes)
        used_values |= candidate_values
        best_benefit = trial_benefit

    return batch, included


def ir_fingerprint(block) -> str:
    """sha1 of the block's printed IR. Same fingerprint, same target and cap give
    the same search result."""
    return hashlib.sha1(str(block.parent_op()).encode()).hexdigest()


def _result(mapper, hardware: HardwareMapping, sink, array_counts: HardwareArrayCounts) -> SearchResult:
    return SearchResult(
        latency_ns=mapper.specs[mapper.node_index[sink]].end,
        array_counts=array_counts,
        mapper=mapper,
        hardware=hardware,
        stationary_placements=hardware.rebuild(),
        pf={mapper.node_order[idx]: pf for idx, pf in mapper.mapped_pf.items()},
    )


def replay_search_state(target: SingleModuleTarget, block, pf_overrides: dict) -> SearchResult:
    """The result of a search that converged to `pf_overrides`, rebuilt without
    searching."""
    mapper, hardware = initial_mapping(target, block)
    mapper.pf_overrides = dict(pf_overrides)
    mapper.compute_all()
    array_counts = rebuild_hardware(hardware)
    sink = next(op for op in block.ops if _is_return(op)).operands[0]
    return _result(mapper, hardware, sink, array_counts)


def _memo_candidates(mapper, memo: SearchMemo, name: str, generate) -> list:
    """generate(), memoized on the current state. Returns fresh copies."""
    if mapper.memo_state is None:
        return generate()
    key = (mapper.memo_state, name)
    cached = memo.candidates.get(key)
    if cached is None:
        cached = memo.candidates[key] = generate()
    return [dict(changes) for changes in cached]


def _apply(mapper, hardware: HardwareMapping, changes: dict) -> HardwareArrayCounts:
    for value, pf in changes.items():
        mapper.pf_overrides[mapper.node_index[value]] = pf
    mapper.compute_all()
    return rebuild_hardware(hardware)


# Memo caches carried by `shared_caches` from one search to the next on the same
# block: each is keyed on all of its inputs, so its entries stay valid.
_SHARED_MAPPER_CACHES = (
    "_spec_cache", "_spec_intern", "_operand_idx_cache", "_pf_cap_cache", "_numel_cache", "_shape_cache",
    "_measure_memo",
)
_SHARED_HARDWARE_CACHES = (
    "_op_mapping_cache", "_placement_cache", "_bin_pack_cache", "_matvec_shape_cache", "_stationary_layout_cache",
)
_SHARED_MEMO_CACHES = ("array_counts", "candidates", "accel_area")


def _adopt_shared_caches(obj, names, shared: dict, prefix: str) -> None:
    for name in names:
        key = prefix + name
        fresh = getattr(obj, name)
        existing = shared.get(key)
        if existing is None:
            shared[key] = fresh
        else:
            existing.update(fresh)
            setattr(obj, name, existing)


def _require_start_fits(mapper, array_counts: HardwareArrayCounts, budget: Budget, accel_index: tuple) -> None:
    """Raises if the mapping the search starts from is already over the budget: the
    search only raises pf, so it could never bring it back within the budget."""
    over = _arrays_over_budget(array_counts, budget)
    if over is not None:
        raise ValueError(f"starting mapping needs {over}")
    kinds = tuple(budget.accel_area_caps)
    if not kinds:
        return
    areas = accel_kinds_peak_area_um2(accel_index, mapper.specs, mapper.mapped_pf, kinds)
    for kind, cap_um2 in budget.accel_area_caps.items():
        if areas[kind] > cap_um2:
            raise ValueError(f"starting mapping needs {areas[kind]:.0f} um^2 of {kind}, over the cap of {cap_um2:.0f}")


def run_capped_search_sync_group(target: SingleModuleTarget | DualModuleTarget, block,
                                 budget: Budget | DualModuleBudget | EnsembleBudget,
                                 warm_start_pf: dict[SSAValue, int],
                                 shared_caches: dict, verbose: bool) -> SearchResult:
    """Raises pf round by round while it shortens the sink's latency and fits
    `budget`. Each round scores per-op, critical-path and sync-group candidates and
    applies the best disjoint batch (_pick_batch); after a round that used a group
    candidate, it keeps escalating groups only (drain) until none helps.

    warm_start_pf   {value: pf} to start from ({} = every op at pf 1).
    shared_caches   memo caches shared across searches on the same block ({} = fresh).
    The non-critical ops are maxed out before the search (_maxout_noncritical_ops) and
    lowered after it to what keeps up with the slower of their producers and consumers
    (_lower_noncritical_pf_to_neighbor_rate).
    """
    t0 = time.time()
    mapper, hardware = initial_mapping(target, block)
    memo = SearchMemo()
    _adopt_shared_caches(mapper, _SHARED_MAPPER_CACHES, shared_caches, "mapper.")
    _adopt_shared_caches(hardware, _SHARED_HARDWARE_CACHES, shared_caches, "hardware.")
    _adopt_shared_caches(memo, _SHARED_MEMO_CACHES, shared_caches, "memo.")
    mapper.pf_overrides.update({mapper.node_index[v]: pf for v, pf in warm_start_pf.items()})
    sync = annotate_syncs(mapper)
    live_groups = _build_live_groups(mapper, sync)
    _maxout_noncritical_ops(mapper, budget.accel_area_caps)
    array_counts = rebuild_hardware(hardware)
    accel_index = accel_area_index(block, mapper.node_index)
    _require_start_fits(mapper, array_counts, budget, accel_index)
    mapper.set_memo(True)

    sink = next(op for op in block.ops if _is_return(op)).operands[0]

    def _round(candidate_sets):
        """Usable candidates of the current state, scored."""
        baseline_latency = mapper.specs[mapper.node_index[sink]].end
        return _usable_candidates(candidate_sets, mapper, hardware, memo, sink, baseline_latency,
                                  array_counts.total_array_count, budget, accel_index)

    it = 0
    while True:
        it += 1
        candidate_sets = _memo_candidates(mapper, memo, "pf", lambda: find_pf_candidates(mapper, sync))
        candidate_sets += _memo_candidates(mapper, memo, "critical_path",
                                           lambda: find_critical_path_candidates(mapper, sink))
        group_candidates = _find_sync_group_candidates(mapper, hardware, memo, live_groups,
                                                       array_counts.total_array_count)
        candidate_sets += group_candidates
        usable = _round(candidate_sets)

        if verbose:
            print(f"  [search] iter {it}: candidates={len(candidate_sets)} usable={len(usable)} "
                  f"elapsed={time.time() - t0:.1f}s", flush=True)
        if not usable:
            break

        best_changes, included = _pick_batch(usable, mapper, hardware, memo, sink, budget, accel_index)
        array_counts = _apply(mapper, hardware, best_changes)

        if any(gc is item for gc in group_candidates for item in included):
            drain_it = 0
            while True:
                drain_candidates = _find_sync_group_candidates(mapper, hardware, memo, live_groups,
                                                               array_counts.total_array_count)
                if not drain_candidates:
                    break
                drain_usable = _round(drain_candidates)
                if not drain_usable:
                    break
                drain_best, _drain_included = _pick_batch(drain_usable, mapper, hardware, memo, sink, budget,
                                                          accel_index)
                array_counts = _apply(mapper, hardware, drain_best)
                drain_it += 1
                if verbose:
                    print(f"    [search drain] +{drain_it} elapsed={time.time() - t0:.1f}s", flush=True)

    # the non-critical ops were maxed out before the search: lower them to what keeps
    # up with the slower of their producers and consumers
    _lower_noncritical_pf_to_neighbor_rate(mapper, hardware, memo, sink, budget, accel_index)
    mapper.compute_all()
    array_counts = rebuild_hardware(hardware)

    # the returned mapper may be changed by the caller
    mapper.set_memo(False)
    return _result(mapper, hardware, sink, array_counts)
