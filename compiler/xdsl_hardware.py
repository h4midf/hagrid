"""Array placement used by the search: HardwareMapping.

Given an already-timed block (a TimingOnlyMapping after compute_all), computes how many arrays every op needs and the peak array
usage: stationary (resident) copies, bin-packed into arrays, and the volatile
arrays each matvec / lut_search uses while it runs. No IR is created or changed;
timing data is read from the TimingOnlyMapping the HardwareMapping is built with.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass, field

from xdsl.ir import Operation, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.rewrite_helpers import shape_of
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_timing import TimingOnlyMapping, _defining_op
from targets.target_loader import MemoryUnit, SingleModuleTarget

MAPPED_OPERATION_OPS = (cd.CimMatvecOp, cd.CimLutSearchOp, cd.CimLutSearchVectorOp)
# set for the hot `type(op) in ...` checks (faster than isinstance())
_MAPPED_OPERATION_OPS_SET = frozenset(MAPPED_OPERATION_OPS)

# size bound of the per-op memos; clearing one only costs recomputation
_HW_CACHE_MAX_ENTRIES = 500_000
# size bound of the bin-packing memo: each key holds every placement (4 ints each), so
# it is far bigger per entry than the per-op memos
_BIN_PACK_CACHE_MAX_ENTRIES = 1_000


@dataclass(frozen=True)
class StationaryPlacement:
    """One resident (stationary) value for one consumer: its RMU tiles and how many copies are stored."""
    copies: int
    full_arrays_per_copy: int     # tiles that fill an array's full width
    partial_tile_cols: int        # cell columns of each narrower tile (0 if none)
    partial_tiles_per_copy: int


@dataclass(frozen=True)
class OperationMapping:
    """Arrays and timing of one mapped op (matvec, lut_search, lut_search_vector) at its pf:
    volatile_arrays from startup (its spec's start less its setup: a dynamic-rhs matvec's
    copies are written then) to output_end_time. A dynamic-rhs matvec may also hold a copy
    of its rhs (rhs_hold_arrays) that the rhs's producer writes as it comes out, from
    rhs_hold_start (the rhs's first batch) to rhs_hold_end (HardwareMapping.HELD_RHS_UNTIL):
    on the coupled RMU/WMU architecture a staging copy in WMUs, until the matvec's spec
    start (its setup has written its copies from it); on a dual-module target the rhs's
    only copy, in the digital module, until the matvec's end. 0 for every other op."""
    pf: int
    startup: float
    output_end_time: float
    volatile_arrays: int
    rhs_hold_arrays: int = field(default=0, kw_only=True)
    rhs_hold_start: float = field(default=0.0, kw_only=True)
    rhs_hold_end: float = field(default=0.0, kw_only=True)


@dataclass(frozen=True)
class VolatileDemand:
    """Arrays an op occupies while it runs, from live_start to live_end."""
    array_count: int
    live_start: float
    live_end: float


@dataclass(frozen=True)
class HardwareArrayCounts:
    """Array totals of a mapping: total = max(stationary, peak volatile)."""
    total_array_count: int
    nonvolatile_array_count: int
    volatile_array_count: int


def _fx_name(op: Operation) -> str:
    """The op's fx_name attribute."""
    return op.attributes["fx_name"].data


class HardwareMapping:
    """Computes the array placement of one already-timed block (see the module docstring).
    Bound to that block and its TimingOnlyMapping at construction; rebuild /
    rebuild_operations read the mapper's current timing (compute_all must have run),
    never write it.
    """

    def __init__(self, target: SingleModuleTarget, block, mapper: TimingOnlyMapping):
        self.architecture = target.architecture
        self._init_units(target)
        self._stationary_layout_cache: dict = {}
        self._matvec_shape_cache: dict = {}
        # per-op memos of rebuild_operations / rebuild
        self._op_mapping_cache: dict = {}
        self._placement_cache: dict = {}
        # memo of _bin_pack_stationary_arrays
        self._bin_pack_cache: dict = {}
        # id(OperationMapping) -> (mapping, [VolatileDemand])
        self._volatile_demand_cache: dict = {}

        self._mapper = mapper
        # (stationary value, consumer) pairs and mapped ops: fixed IR structure, built once
        self._stationary_pairs: list = []
        for op in block.ops:
            if type(op) is not cd.CimSaveStationaryOp:
                continue
            stat_value = op.results[0]
            for consumer_op in self.stationary_consumers(stat_value):
                self._stationary_pairs.append((stat_value, consumer_op, mapper.node_index[stat_value],
                                               mapper.node_index[consumer_op.results[0]]))
        self._mapped_ops: list = [
            (op, op.results[0], mapper.node_index[op.results[0]])
            for op in block.ops if type(op) in _MAPPED_OPERATION_OPS_SET
        ]
        # matvecs that hold a copy of their dynamic rhs from its first batch
        # (OperationMapping.rhs_hold_*): op position -> its rhs's position
        self._held_rhs: dict[int, int] = {
            idx: mapper.node_index[op.rhs] for op, _value, idx in self._mapped_ops if self._holds_rhs(op)}

    # ---- a dynamic rhs held from its first batch (OperationMapping.rhs_hold_*)
    # until the matvec's "start" (staged, then copied by its setup) or "end" (its only copy)
    HELD_RHS_UNTIL = "start"

    def _holds_rhs(self, op: Operation) -> bool:
        """On the coupled RMU/WMU architecture, a dynamic-rhs matvec stages its rhs in WMUs."""
        return (self.architecture == "coupled_fine_grained_rmu_wmu" and type(op) is cd.CimMatvecOp
                and not self._is_stationary_value(op.rhs))

    def _held_rhs_arrays(self, rhs: SSAValue) -> int:
        """Arrays of the held copy: one rhs copy in WMUs."""
        return self._matvec_matrix_array_requirement(rhs)

    def _init_units(self, target: SingleModuleTarget) -> None:
        """The array units: stationary operands (weights) live in the RMU, dynamic
        operands in the WMU, both the same arrays."""
        self.rmu = target.geometry.rmu
        self.wmu = target.geometry.wmu

    def rebuild(self) -> list[StationaryPlacement]:
        """Stationary placements of every (stationary value, consumer) pair, at the current pf."""
        mapper = self._mapper
        placements = []
        for stat_value, consumer_op, stat_idx, consumer_idx in self._stationary_pairs:
            # a placement depends only on the consumer's pf (copies); its tiles are fixed
            key = (stat_idx, consumer_idx, mapper.mapped_pf[consumer_idx])
            cached = self._placement_cache.get(key)
            if cached is not None:
                placements.append(cached)
                continue
            # not cached: compute the placement and store it (placements are immutable)
            full_arrays, partial_cols, partial_tiles = self._stationary_layout(stat_value)
            entry = StationaryPlacement(
                copies=self.stationary_copies_for_consumer(consumer_op),
                full_arrays_per_copy=full_arrays,
                partial_tile_cols=partial_cols,
                partial_tiles_per_copy=partial_tiles,
            )
            if len(self._placement_cache) >= _HW_CACHE_MAX_ENTRIES:
                self._placement_cache.clear()
            self._placement_cache[key] = entry
            placements.append(entry)

        return placements

    # ---- the pools of volatile arrays, for array ids (xdsl_materialize)
    def volatile_pool(self, value: SSAValue) -> str:
        """The pool a mapped op's volatile arrays are in: one pool of arrays here."""
        return "all"

    def volatile_pools(self, counts: HardwareArrayCounts) -> list[tuple[str, int, int]]:
        """(pool, peak volatile arrays, total arrays) of every pool, in id order: a pool's
        ids follow the previous pools' arrays."""
        return [("all", counts.volatile_array_count, counts.total_array_count)]

    def held_rhs(self) -> list[tuple[int, int, int]]:
        """(op position, rhs position, arrays) of every matvec holding a copy of its rhs
        from the rhs's first batch to its HELD_RHS_UNTIL (OperationMapping.rhs_hold_*);
        the arrays do not depend on pf."""
        return [(idx, rhs_idx, self._held_rhs_arrays(self._mapper.node_order[rhs_idx]))
                for idx, rhs_idx in self._held_rhs.items()]

    def stationary_placements_by_pair(self) -> list[tuple[SSAValue, Operation, StationaryPlacement]]:
        """(stationary value, consumer, placement) of every pair, at the current pf."""
        return [(stat_value, consumer_op, placement)
                for (stat_value, consumer_op, _, _), placement in zip(self._stationary_pairs, self.rebuild())]

    def rebuild_operations(self) -> dict[SSAValue, OperationMapping]:
        """OperationMapping of every mapped op at the current pf."""
        mapper = self._mapper
        mappings = {}
        for op, value, idx in self._mapped_ops:
            # an op's mapping depends on its pf and its spec (and a held rhs's spec);
            # specs are immutable and interned, so the same object means the same values
            rhs_idx = self._held_rhs.get(idx)
            key = (idx, mapper.mapped_pf[idx], mapper.specs[idx],
                   None if rhs_idx is None else mapper.specs[rhs_idx])
            cached = self._op_mapping_cache.get(key)
            if cached is not None:
                mappings[value] = cached
                continue
            result = self.map_operation(op)
            if len(self._op_mapping_cache) >= _HW_CACHE_MAX_ENTRIES:
                self._op_mapping_cache.clear()
                self._volatile_demand_cache.clear()
            self._op_mapping_cache[key] = result
            mappings[value] = result
        return mappings

    def array_counts(
        self,
        stationary_placements: list[StationaryPlacement],
        operation_mappings: dict[SSAValue, OperationMapping],
    ) -> HardwareArrayCounts:
        """Array totals: stationary copies bin-packed into arrays, plus the peak of the
        volatile demands over time.
        """
        nonvolatile_array_count = self._bin_pack_stationary_arrays(stationary_placements, self.rmu.cols)
        # VolatileDemands per OperationMapping object, built once
        demand_cache = self._volatile_demand_cache
        volatile_demands = []
        for m in operation_mappings.values():
            entry = demand_cache.get(id(m))
            if entry is None or entry[0] is not m:
                # the op's arrays, and its staged rhs (both while the setup copies it)
                demands = []
                if m.volatile_arrays > 0:
                    demands.append(VolatileDemand(
                        array_count=m.volatile_arrays,
                        live_start=m.startup, live_end=m.output_end_time,
                    ))
                if m.rhs_hold_arrays > 0:
                    demands.append(VolatileDemand(
                        array_count=m.rhs_hold_arrays,
                        live_start=m.rhs_hold_start, live_end=m.rhs_hold_end,
                    ))
                entry = demand_cache[id(m)] = (m, demands)
            volatile_demands.extend(entry[1])
        volatile_array_count = self._peak_volatile_array_count(volatile_demands)
        return HardwareArrayCounts(
            total_array_count=max(nonvolatile_array_count, volatile_array_count),
            nonvolatile_array_count=nonvolatile_array_count,
            volatile_array_count=volatile_array_count,
        )

    def _bin_pack_stationary_arrays(self, stationary_placements: list[StationaryPlacement], capacity: int) -> int:
        """_bin_pack_stationary_arrays_uncached, memoized on the array width and the
        placements' tiles and copies."""
        key = (capacity,) + tuple([(p.full_arrays_per_copy, p.partial_tile_cols, p.partial_tiles_per_copy, p.copies)
                                   for p in stationary_placements])
        cached = self._bin_pack_cache.get(key)
        if cached is not None:
            return cached
        result = self._bin_pack_stationary_arrays_uncached(stationary_placements, capacity)
        if len(self._bin_pack_cache) >= _BIN_PACK_CACHE_MAX_ENTRIES:
            self._bin_pack_cache.clear()
        self._bin_pack_cache[key] = result
        return result

    def _bin_pack_stationary_arrays_uncached(self, stationary_placements: list[StationaryPlacement],
                                             capacity: int, opened: list | None = None) -> int:
        """Best-fit column-strip packing of every stationary copy's tiles into arrays of
        `capacity` cell columns each. A full-width tile takes a whole array. A narrower tile takes
        its columns over the full array height and can go into the leftover columns of
        another value's array; tiles of the same placement never share an array (they
        compute at the same time, and tiles in one array share its wordlines).
        Free space is tracked as free columns -> number of arrays, so the cost grows with the
        number of placements, not with the number of tiles.
        `opened`, if given, gets one entry per placement: the arrays that placement
        opened (a tile put into another array's free columns opens none), so the
        entries sum to the result.
        """
        bin_count = 0
        pool_keys: list[int] = []
        pool_counts: dict[int, int] = {}

        for placement in stationary_placements:
            count_before = bin_count
            bin_count += placement.full_arrays_per_copy * placement.copies
            size = placement.partial_tile_cols
            tiles = placement.partial_tiles_per_copy * placement.copies

            own_new_counts: dict[int, int] = {}

            idx = bisect.bisect_left(pool_keys, size)
            take_remaining = tiles
            consumed_upto = idx
            while take_remaining > 0 and consumed_upto < len(pool_keys):
                key = pool_keys[consumed_upto]
                avail = pool_counts[key]
                take = min(avail, take_remaining)
                take_remaining -= take
                remainder = key - size
                if remainder > 0:
                    own_new_counts[remainder] = own_new_counts.get(remainder, 0) + take
                if take == avail:
                    del pool_counts[key]
                    consumed_upto += 1
                else:
                    pool_counts[key] = avail - take
            if consumed_upto > idx:
                del pool_keys[idx:consumed_upto]

            if take_remaining > 0:
                bin_count += take_remaining
                own_new_counts[capacity - size] = own_new_counts.get(capacity - size, 0) + take_remaining

            for key, count in own_new_counts.items():
                if key in pool_counts:
                    pool_counts[key] += count
                else:
                    pool_counts[key] = count
                    bisect.insort(pool_keys, key)
            if opened is not None:
                opened.append(bin_count - count_before)

        return bin_count

    def _peak_volatile_array_count(self, volatile_demands: list[VolatileDemand]) -> int:
        """Peak number of volatile arrays in use at the same time."""
        events = []
        for demand in volatile_demands:
            events.append((demand.live_start, 0, demand.array_count))
            events.append((demand.live_end, 1, demand.array_count))

        events.sort()
        active = 0
        peak = 0
        for _time, kind, count in events:
            if kind == 1:
                active -= count
            else:
                active += count
                peak = max(peak, active)
        return peak

    def map_operation(self, op: Operation) -> OperationMapping:
        """OperationMapping of one mapped op at its current pf."""
        pf = self._operation_pf(op.results[0])
        return self._operation_mapping_from_counts(op, pf=pf, volatile_arrays=self.volatile_arrays_at_pf(op, pf))

    def volatile_arrays_at_pf(self, op: Operation, pf: int) -> int:
        """Arrays a mapped op (matvec, lut_search, lut_search_vector) occupies while it
        runs, at this pf."""
        if type(op) is cd.CimMatvecOp:
            return self._matvec_volatile_arrays(op, pf)
        if type(op) is cd.CimLutSearchOp:
            return self._lut_volatile_arrays(op.table, "lut_search", pf)
        if type(op) is cd.CimLutSearchVectorOp:
            return self._lut_volatile_arrays(op.lhs, "lut_search_vector", pf)
        raise RuntimeError(f"hardware operation mapping: unsupported op {op}")

    def _matvec_volatile_arrays(self, op: cd.CimMatvecOp, pf: int) -> int:
        """Matvec with a stationary rhs (weights) or with both operands dynamic."""
        lhs, rhs = op.lhs, op.rhs
        lhs_stat = self._is_stationary_value(lhs)
        rhs_stat = self._is_stationary_value(rhs)
        if lhs_stat:
            raise RuntimeError("hardware matvec mapping: first operand stationary is unsupported")
        if rhs_stat:
            return self._matvec_stationary_second_arrays(op, rhs, lhs, pf)
        return self._matvec_both_dynamic_arrays(op, lhs, rhs, pf)

    def _matvec_stationary_second_arrays(self, op: cd.CimMatvecOp, stat_value: SSAValue, dyn_value: SSAValue,
                                         pf: int) -> int:
        """Matvec with stationary weights. The MAC happens in the RMU arrays that store the
        weights, so the arrays the op occupies are its resident copies (one per parallel
        copy, stationary_copies_at_pf).
        """
        if not self._is_grouped_stationary_matvec(op, stat_value, dyn_value):
            out_dim = self._matvec_out_dim(op)
            in_dim, stat_out_dim = self._matvec_stationary_dims(stat_value)
            if out_dim != stat_out_dim:
                raise RuntimeError("hardware matvec mapping: output last dim does not match stationary out dim")
            self._require_input_last_dim(dyn_value, in_dim)
        return self.stationary_arrays_per_copy(stat_value) * self.stationary_copies_at_pf(pf)

    def _matvec_both_dynamic_arrays(self, op: cd.CimMatvecOp, lhs: SSAValue, rhs_value: SSAValue, pf: int) -> int:
        """Matvec with a dynamic rhs: MAC from the WMU. Each parallel copy holds the rhs
        twice in the WMU: a compute copy (destroyed by every WMU MAC) and a keep copy
        it is rewritten from.
        """
        if self.architecture != "coupled_fine_grained_rmu_wmu":
            raise NotImplementedError(f"matvec with a dynamic rhs: not modeled for {self.architecture}")
        out_dim = self._matvec_out_dim(op)
        in_dim, rhs_out_dim = self._matvec_dynamic_rhs_dims(rhs_value)
        if out_dim != rhs_out_dim:
            raise RuntimeError("hardware matvec mapping: output last dim does not match rhs out dim")
        self._require_input_last_dim(lhs, in_dim)
        return 2 * self._matvec_matrix_array_requirement(rhs_value) * pf

    def _lut_volatile_arrays(self, lut_value: SSAValue, name: str, pf: int) -> int:
        """lut_search (one value per index) and lut_search_vector (a whole row per
        index): one LUT copy per parallel lookup."""
        if not self._is_stationary_value(lut_value):
            raise RuntimeError(f"hardware {name} mapping: LUT arg must be save_stationary")
        return self.stationary_arrays_per_copy(lut_value) * pf

    def _operation_mapping_from_counts(self, op: Operation, *, pf: int, volatile_arrays: int) -> OperationMapping:
        """Builds the OperationMapping from the array count and the op's timing spec, with
        the staged rhs copy of a matvec in _held_rhs."""
        value = op.results[0]
        idx = self._mapper.node_index[value]
        start = self._required_time(value, "start")
        hold = {}
        if idx in self._held_rhs:
            hold = {"rhs_hold_arrays": self._held_rhs_arrays(op.rhs),
                    "rhs_hold_start": self._required_time(op.rhs, "start"),
                    "rhs_hold_end": self._required_time(value, self.HELD_RHS_UNTIL)}
        return OperationMapping(
            pf=pf,
            startup=start - self._mapper.setup_at(idx),
            output_end_time=self._required_time(value, "end"),
            volatile_arrays=volatile_arrays,
            **hold,
        )

    def _operation_pf(self, value: SSAValue) -> int:
        """pf of an op: the mapper's mapped_pf, the same capped pf its timing used."""
        return self._mapper.mapped_pf[self._mapper.node_index[value]]

    def _required_time(self, value: SSAValue, field: str) -> float:
        """A field (start / end) of a value's timing spec."""
        return float(getattr(self._mapper.specs[self._mapper.node_index[value]], field))

    def _matvec_out_dim(self, op: cd.CimMatvecOp) -> int:
        """Output width (last dim) of a matvec."""
        out_shape = shape_of(op.results[0].type)
        if len(out_shape) < 1:
            raise RuntimeError("hardware matvec mapping: invalid output rank")
        return out_shape[-1]

    def _matvec_matrix_array_requirement(self, value: SSAValue) -> int:
        """WMU arrays one copy of a dynamic matvec rhs needs."""
        key = ("matrix_array_requirement", value)
        cached = self._matvec_shape_cache.get(key)
        if cached is not None:
            return cached
        shape = shape_of(value.type)
        in_dim, out_dim = self._matvec_dynamic_rhs_dims(value)
        blocks = 1
        for d in shape[:-2]:
            blocks *= d
        full_arrays, _partial_cols, partial_tiles = self._tile_layout(
            blocks, in_dim, out_dim, bit_width(value), self.wmu)
        result = full_arrays + partial_tiles
        self._matvec_shape_cache[key] = result
        return result

    def _is_grouped_stationary_matvec(self, op: cd.CimMatvecOp, stat_value: SSAValue, dyn_value: SSAValue) -> bool:
        """True for a grouped matvec (3D stationary [nb, v, k]); raises if its lhs/output
        shapes do not match the groups.
        """
        stat_shape = shape_of(stat_value.type)
        if len(stat_shape) != 3:
            return False
        dyn_shape = shape_of(dyn_value.type)
        out_shape = shape_of(op.results[0].type)
        if len(dyn_shape) < 2 or len(out_shape) < 2:
            raise RuntimeError("hardware matvec mapping: grouped matvec expects lhs/out rank >= 2")
        nb, v, k = stat_shape[0], stat_shape[1], stat_shape[2]
        if dyn_shape[-2:] != [nb, v]:
            raise RuntimeError(f"hardware matvec mapping: grouped lhs trailing dims {dyn_shape[-2:]} do not match [{nb}, {v}]")
        if out_shape[-2:] != [nb, k]:
            raise RuntimeError(f"hardware matvec mapping: grouped output trailing dims {out_shape[-2:]} do not match [{nb}, {k}]")
        return True

    def _matvec_stationary_dims(self, stat_value: SSAValue) -> tuple[int, int]:
        """(in, out) dims of a 2D stationary matrix."""
        key = ("stationary_dims", stat_value)
        cached = self._matvec_shape_cache.get(key)
        if cached is not None:
            return cached
        shape = shape_of(stat_value.type)
        if len(shape) != 2:
            raise RuntimeError(f"hardware matvec mapping: expected 2D stationary, got {shape}")
        result = (shape[0], shape[1])
        self._matvec_shape_cache[key] = result
        return result

    def _matvec_dynamic_rhs_dims(self, rhs_value: SSAValue) -> tuple[int, int]:
        """(in, out) dims of a dynamic matvec rhs [..., in, out] (lhs [..., in] @ rhs)."""
        key = ("dynamic_rhs_dims", rhs_value)
        cached = self._matvec_shape_cache.get(key)
        if cached is not None:
            return cached
        shape = shape_of(rhs_value.type)
        if len(shape) < 2:
            raise RuntimeError(f"hardware matvec mapping: invalid rhs rank {shape}")
        result = (shape[-2], shape[-1])
        self._matvec_shape_cache[key] = result
        return result

    def _require_input_last_dim(self, value: SSAValue, expected_dim: int):
        """Raises if value's last dim is not expected_dim."""
        shape = shape_of(value.type)
        if len(shape) < 1 or shape[-1] != expected_dim:
            raise RuntimeError(f"hardware matvec mapping: dynamic last dim does not match expected dim {expected_dim}")

    def _is_stationary_value(self, value: SSAValue) -> bool:
        """True if value comes from a save_stationary."""
        return type(_defining_op(value)) is cd.CimSaveStationaryOp

    def stationary_consumers(self, stat_value: SSAValue) -> list[Operation]:
        """Consumers of a stationary value that place it (MAPPED_OPERATION_OPS), sorted by name."""
        consumers: dict[int, Operation] = {}
        for use in stat_value.uses:
            op = use.operation
            if type(op) not in _MAPPED_OPERATION_OPS_SET:
                continue
            consumers[id(op)] = op
        return sorted(consumers.values(), key=_fx_name)

    def _stationary_layout(self, stat_value: SSAValue) -> tuple[int, int, int]:
        """RMU tiles of one resident copy (_tile_layout). A value [..., in, out] is
        prod(...) matrices [in, out]; a 1D LUT table [n] is one column of n entries. A
        grouped matvec's weight [g, v, k] whose groups fit an array is laid out block-
        diagonally (_block_diagonal_layout).
        """
        shape = tuple(shape_of(stat_value.type))
        bits = bit_width(stat_value)
        cache_key = (shape, bits)
        cached = self._stationary_layout_cache.get(cache_key)
        if cached is not None:
            return cached
        if len(shape) == 3 and self._groups_per_array(shape[1], shape[2], bits, self.rmu) > 0:
            result = self._block_diagonal_layout(*shape, bits, self.rmu)
        else:
            if len(shape) == 1:
                blocks, in_dim, out_dim = 1, shape[0], 1
            else:
                blocks = 1
                for d in shape[:-2]:
                    blocks *= d
                in_dim, out_dim = shape[-2], shape[-1]
            result = self._tile_layout(blocks, in_dim, out_dim, bits, self.rmu)
        self._stationary_layout_cache[cache_key] = result
        return result

    def _groups_per_array(self, v: int, k: int, bits: int, unit: MemoryUnit) -> int:
        """Groups of a grouped weight [g, v, k] one array holds block-diagonally: each in
        its own v rows and its own k * cells columns. 0 if one group does not fit."""
        return min(unit.rows // v, unit.cols // (k * self._cells_per_element(bits, unit)))

    def _block_diagonal_layout(self, g: int, v: int, k: int, bits: int, unit: MemoryUnit) -> tuple[int, int, int]:
        """Tiles of a grouped weight [g, v, k] laid out block-diagonally: _groups_per_array
        groups per array, each group in its own rows and columns, so one array operation
        computes all of them (the input carries each group's v values in its own rows;
        every other cell of a group's columns holds 0). Returns _tile_layout's (full-width
        tiles, cell columns of each narrower tile, number of narrower tiles); the last
        array may hold fewer groups and is counted at the same width."""
        per_array = self._groups_per_array(v, k, bits, unit)
        arrays = -(-g // per_array)
        width = per_array * k * self._cells_per_element(bits, unit)
        if width == unit.cols:
            return arrays, 0, 0
        return 0, width, arrays

    def stationary_arrays_per_copy(self, stat_value: SSAValue) -> int:
        """Arrays one resident copy needs on its own."""
        full_arrays, _partial_cols, partial_tiles = self._stationary_layout(stat_value)
        return full_arrays + partial_tiles

    def _tile_layout(self, blocks: int, in_dim: int, out_dim: int, bits: int, unit: MemoryUnit) -> tuple[int, int, int]:
        """Tiles of `blocks` matrices [in_dim, out_dim] in `unit`: in_dim on the rows,
        out_dim * cells per element on the columns. Returns (full-width tiles, cell
        columns of each narrower tile, number of narrower tiles).
        """
        row_tiles = -(-in_dim // unit.rows)
        cols = out_dim * self._cells_per_element(bits, unit)
        full_col_tiles, partial_cols = divmod(cols, unit.cols)
        strips = blocks * row_tiles
        return strips * full_col_tiles, partial_cols, (strips if partial_cols else 0)

    def _cells_per_element(self, bits: int, unit: MemoryUnit) -> int:
        """Cells one element of `bits` bits occupies in `unit`."""
        return -(-bits // unit.bits_per_cell)

    def stationary_copies_for_consumer(self, consumer_op: Operation) -> int:
        """Resident copies a consumer needs at its current pf (stationary_copies_at_pf)."""
        return self.stationary_copies_at_pf(self._operation_pf(consumer_op.results[0]))

    def stationary_copies_at_pf(self, pf: int) -> int:
        """Resident copies a consumer at this pf needs: one per parallel copy. A stored
        copy computes in its own RMU array, so it cannot feed several parallel copies.
        Storing extra copies is ordinary capacity, limited only by the array cap.
        """
        return pf

    def stationary_pairs(self) -> list[tuple[SSAValue, Operation]]:
        """Every (stationary value, consumer) pair, in rebuild's order."""
        return [(stat_value, consumer_op) for stat_value, consumer_op, _, _ in self._stationary_pairs]

    def stationary_live_start(self, stat_value: SSAValue) -> float:
        """When a resident copy becomes live: when the stationary value is ready."""
        return self._required_time(stat_value, "end")

    def stationary_live_end(self, consumer_op: Operation) -> float:
        """When a resident copy is no longer needed: when its consumer ends."""
        return self._required_time(consumer_op.results[0], "end")
