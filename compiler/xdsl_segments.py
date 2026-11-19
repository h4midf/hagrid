"""Splits a tiled cim block into segments: maximal runs of a repeated op sequence (a
transformer layer, x12) and the residual chunks between them. A search solves each
segment once, on a standalone copy of its first occurrence, and its pf assignment is
replicated onto every occurrence.

Segments come from xdsl_auto_decompose.find_repeat_segments (op class, result type
and pf cap per op) and are cut out with extract_region (a clone of the ops; inputs
from before the segment become block arguments). find_segments then requires every
occurrence to have the same timing as the solved copy, op by op: the same node form
and, at pf 1 and at the pf cap, the same recipe up to which nodes its operands are.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Block, SSAValue

from compiler.xdsl_auto_decompose import extract_region, find_repeat_segments
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_layer_decompose import _node_desc, _real_values
from compiler.xdsl_search import initial_mapping

_MAPPED_OPERATION_TYPES = frozenset(MAPPED_OPERATION_OPS)
from compiler.xdsl_timing import TimingOnlyMapping


@dataclass
class Segment:
    """One segment: a standalone block (a copy of the first occurrence) and, per
    occurrence in the full block, the full block's values in the same order as
    region_values (position j of every occurrence corresponds to region_values[j])."""
    block: Block
    region_values: list[SSAValue]
    occurrences: list[list[SSAValue]]

    @property
    def repeats(self) -> int:
        return len(self.occurrences)


def _timing_signature(mapper: TimingOnlyMapping, value: SSAValue) -> tuple:
    """The node's timing, without which nodes its operands are."""
    idx = mapper.node_index[value]
    kind, arg = mapper.node_form(idx)
    if kind == "trivial":
        return (kind,)
    if kind == "alias":
        return (kind,)
    if kind == "barrier":
        return (kind, tuple(n for _p, n in arg))
    sig = [kind]
    for pf in sorted({1, mapper.pf_cap_at(idx)}):
        r = arg(pf)
        sig.append((pf, tuple(n for _p, n in r.first_needs), tuple(n for _p, n in r.all_needs),
                    tuple(t[1:] for t in r.batch_terms), r.num_problems, r.outputs_per_problem,
                    r.hardware_interval, r.pf, r.setup))
    return tuple(sig)


def _build_segment(target, mapper: TimingOnlyMapping, values: list, start: int, length: int, reps: int):
    """(Segment, None), or (None, k) if occurrence k has an op whose timing differs from
    the solved copy's. That happens when a cut separates an op from a producer its
    timing depends on (a reduce_vec summing a matvec's row tiles: in the copy, its
    operand is a block argument, so it is timed as a plain reduction)."""
    occurrences = [values[start + k * length: start + (k + 1) * length] for k in range(reps)]
    region_block, region_values = extract_region(occurrences[0])
    # the target's own timing model (TimingOnlyMapping, DualModuleTimingMapping, ...)
    region_mapper, _hardware = initial_mapping(target, region_block)
    expected = [_timing_signature(region_mapper, v) for v in region_values]
    for k, occ in enumerate(occurrences):
        if any(_timing_signature(mapper, v) != expected[j] for j, v in enumerate(occ)):
            return None, k
    return Segment(block=region_block, region_values=region_values, occurrences=occurrences), None


def _merge_residuals(spans: list) -> list:
    """Adjacent residual spans (repeat count 1) joined into one."""
    merged = []
    for start, length, reps in spans:
        if reps == 1 and merged and merged[-1][2] == 1:
            merged[-1] = (merged[-1][0], merged[-1][1] + length, 1)
        else:
            merged.append((start, length, reps))
    return merged


def find_segments(target, block, mapper: TimingOnlyMapping) -> list[Segment]:
    """The block's segments, in order; together they cover every value of
    _real_values(mapper) exactly once. `mapper` is a TimingOnlyMapping of `block`.

    A segment whose first occurrence's timing differs from the full block's is merged
    with the segment before it (the cut before it is wrong); one whose later occurrence
    differs becomes residual (the cut between its repeats is wrong). Repeated until
    every segment's timing matches."""
    values = _real_values(mapper)
    descs = [_node_desc(mapper, v) for v in values]
    spans = list(find_repeat_segments(descs))
    built: dict = {}
    while True:
        segments, bad = [], None
        for s, span in enumerate(spans):
            if span not in built:
                built[span] = _build_segment(target, mapper, values, *span)
            seg, failed_occurrence = built[span]
            if seg is None:
                bad = (s, failed_occurrence)
                break
            segments.append(seg)
        if bad is None:
            break
        s, k = bad
        start, length, reps = spans[s]
        if k == 0:
            if s == 0:
                raise ValueError(f"find_segments: the first segment's timing differs from the full block's")
            prev_start, prev_length, prev_reps = spans[s - 1]
            spans[s - 1:s + 1] = [(prev_start, prev_length * prev_reps + length * reps, 1)]
        else:
            spans[s] = (start, length * reps, 1)
        spans = _merge_residuals(spans)
    covered = sum(len(s.region_values) * s.repeats for s in segments)
    if covered != len(values):
        raise RuntimeError(f"find_segments: segments cover {covered} of {len(values)} ops")
    return segments


@dataclass(frozen=True)
class SegmentStationary:
    """A segment's share of the stationary arrays of a mapping of the full block."""
    per_occurrence: tuple[int, ...]   # arrays each occurrence opened in the full packing
    cap_per_copy: int                 # the cap of one solve of the segment, in the units
                                      # CP-SAT counts (stationary_by_segment)

    @property
    def total(self) -> int:
        return sum(self.per_occurrence)


def stationary_by_segment(segments: list[Segment], hardware) -> list[SegmentStationary]:
    """Splits the stationary arrays of `hardware` (a HardwareMapping of the full block,
    at its mapper's current pf) by segment, exactly: the full block's own packing
    (HardwareMapping._bin_pack_stationary_arrays_uncached) is rerun, recording which
    placement opened each array, and each placement belongs to the occurrence its
    consumer op is in. The per-occurrence counts sum to the full block's stationary
    count (total).

    cap_per_copy is in CP-SAT's units (xdsl_cpsat._solve_segment counts each resident
    tile as its own array): the tiles of the segment's resident copies at the
    mapping's pf, summed over its occurrences // repeats. Not the packed share: a segment whose
    tiles fill columns of arrays another segment opened is credited fewer arrays than
    it has tiles, and could not fit even at the mapping's own pfs. The full block's
    packed count is checked against the cap after the replay (solve_capped)."""
    owner = {}
    for s, seg in enumerate(segments):
        for k, occ in enumerate(seg.occurrences):
            for v in occ:
                owner[v] = (s, k)
    pairs = hardware.stationary_placements_by_pair()
    opened: list[int] = []
    # stationary values live in the RMU (single module) or the analog arrays (dual module)
    cols = hardware.rmu.cols if hardware.rmu is not None else hardware.units["analog"].cols
    full_count = hardware._bin_pack_stationary_arrays_uncached([p for _s, _c, p in pairs], cols, opened=opened)
    arrays: dict = {}
    tiles: dict = {}
    for (stat_value, consumer_op, _placement), n in zip(pairs, opened):
        key = owner[consumer_op.results[0]]
        arrays[key] = arrays.get(key, 0) + n
        tiles[key] = tiles.get(key, 0) + (hardware.stationary_arrays_per_copy(stat_value)
                                          * hardware.stationary_copies_for_consumer(consumer_op))
    result = []
    for s, seg in enumerate(segments):
        per_occ = tuple(arrays.get((s, k), 0) for k in range(seg.repeats))
        # every occurrence gets the same pfs (solve_capped), so one occurrence's share is
        # the mean: the most of any occurrence, times repeats, could exceed the cap
        cap = sum(tiles.get((s, k), 0) for k in range(seg.repeats)) // seg.repeats
        result.append(SegmentStationary(per_occurrence=per_occ, cap_per_copy=cap))
    if sum(r.total for r in result) != full_count:
        raise RuntimeError("stationary_by_segment: the split does not sum to the full block's count")
    return result


def sram_correction_by_segment(segments: list[Segment], hardware) -> list[SegmentStationary]:
    """Splits the SRAM correction macros of `hardware` (an EnsembleHardwareMapping of the
    full block, at its mapper's current pf) by segment: each stored matvec's correction
    (macros_at_pf) belongs to the occurrence it is in. A plain sum per op, so the totals
    sum to the full block's correction exactly."""
    pf_at = hardware._mapper.mapped_pf
    node_index = hardware._mapper.node_index
    result = []
    for seg in segments:
        per_occ = tuple(
            sum(hardware.macros_at_pf(v.owner, pf_at[node_index[v]])[2]
                for v in occ if type(v.owner) in _MAPPED_OPERATION_TYPES)
            for occ in seg.occurrences)
        result.append(SegmentStationary(per_occurrence=per_occ, cap_per_copy=sum(per_occ) // seg.repeats))
    return result


def replicate(segments: list[Segment], segment_pfs: list[dict[SSAValue, int]]) -> dict[SSAValue, int]:
    """The full block's {value: pf}: segment_pfs[s] ({region value: pf}) copied onto
    every occurrence of segment s."""
    full_pf = {}
    for seg, pfs in zip(segments, segment_pfs):
        for occ in seg.occurrences:
            for region_v, full_v in zip(seg.region_values, occ):
                if region_v in pfs:
                    full_pf[full_v] = pfs[region_v]
    return full_pf
