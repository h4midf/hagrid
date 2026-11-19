"""CP-SAT model of the timing model (xdsl_timing.TimingOnlyMapping): the same specs,
as integer constraints, with one mode per candidate pf of each op.

Every node's spec comes from TimingOnlyMapping.node_form, so this module has no
per-op formulas of its own:
  trivial    start = end = 0, an unbounded batch (every element ready at start)
  alias      the spec of another node
  barrier    start = end = when every needed operand element is ready (cat)
  pipelined  _pipelined_spec of the node's PipelineRecipe at its pf

Time is integer: ns * scale (scale = 1000 counts ps). Hardware intervals are rounded
to the time unit. The float model's k-th batch, first + (k - 1) * (end - first) /
(count - 1), is here first + floor((k - 1) * (end - first) / (count - 1)), computed
exactly without the (k - 1) * (end - first) product (see ready). validate_against_mapper measures
the rounding against the float model.

Every variable's domain is a provable bound, not a guessed horizon: a node ends no
later than its slowest mode (smallest pf) after its producers' latest ends, since an
operand's elements are never ready after its producer ends.
"""
import math
from dataclasses import dataclass

from ortools.sat.python import cp_model

from compiler.xdsl_accel_concurrency import accel_area_index
from compiler.xdsl_dual_module_timing import op_module
from compiler.xdsl_hardware import MAPPED_OPERATION_OPS
from compiler.xdsl_search import (
    Budget, DualModuleBudget, EnsembleBudget, SearchMemo, SearchResult, _arrays_over_budget,
    _lower_noncritical_pf_to_neighbor_rate, _maxout_noncritical_ops, _result, initial_mapping, rebuild_hardware,
)
from compiler.xdsl_segments import Segment, SegmentStationary
from compiler.xdsl_timing import TimingOnlyMapping, _defining_op, _is_return
from targets.target_loader import DualModuleTarget, EnsembleAppTarget


class CpsatTiming:
    """The timing constraints of a block at a choice of pf per op.

    pf_modes  {node position: [candidate pf, ...]} for every pipelined node, each pf
              already within the node's cap. One candidate fixes the node's pf.
    tie       {node position: earlier node position}: the node takes the same pf as the
              earlier one (the same mode variables; both need the same candidates).
    scale     time units per ns.

    After construction: start[i] / end[i] (expressions, time units) and end_ub[i] (a
    bound on end[i]) per node, and mode[i] {pf: BoolVar} for every node with more than
    one candidate (exactly one true).
    """

    def __init__(self, mapper: TimingOnlyMapping, pf_modes: dict[int, list[int]], scale: int = 1000,
                 model: cp_model.CpModel | None = None, tie: dict[int, int] | None = None):
        self.mapper = mapper
        self.scale = scale
        self.model = model if model is not None else cp_model.CpModel()
        self.pf_modes = pf_modes
        self.tie = tie or {}
        n = len(mapper.node_order)
        self.kind: list[str | None] = [None] * n
        self.root: list[int] = list(range(n))   # alias nodes point at the node whose spec they share
        self.start: list = [None] * n
        self.end: list = [None] * n
        self.end_ub: list[int] = [0] * n
        self.first: list = [None] * n           # first batch out (pipelined only)
        self.unit: list = [None] * n            # batch size (pipelined only)
        self.count: list = [None] * n           # batches (pipelined only)
        self.total: list = [None] * n           # num_problems * outputs_per_problem (pipelined only)
        self.max_count: list = [None] * n
        self.mode: dict[int, dict[int, cp_model.IntVar]] = {}
        self._ready_cache: dict = {}
        self._aux = 0
        for i in range(n):
            self._build_node(i)

    # -- helpers -------------------------------------------------------------------

    def _var(self, lb: int, ub: int, label: str) -> cp_model.IntVar:
        self._aux += 1
        return self.model.new_int_var(lb, ub, f"{label}_{self._aux}")

    def _ticks(self, ns: float) -> int:
        return int(round(ns * self.scale))

    def by_mode(self, i: int, values: dict[int, int]):
        """An expression equal to values[pf] for the node's chosen pf."""
        pfs = self.pf_modes[i]
        if len(pfs) == 1:
            return values[pfs[0]]
        return sum(self.mode[i][pf] * values[pf] for pf in pfs)

    def _needs_ub(self, needs) -> int:
        return max((self.end_ub[self.root[p]] for p, _n in needs), default=0)

    def ready(self, p: int, n: int):
        """When the first n elements of node p are available (_time_for_n_elements):
        first + floor(j * span / d), j = k - 1 for k = min(ceil(n / unit), count)
        batches, span = end - first, d = max(count - 1, 1) (j = 0 when count = 1). With
        span = I * d + rem (0 <= rem < d) this is first + j * I + floor(j * rem / d):
        every product stays within span or count^2."""
        p = self.root[p]
        kind = self.kind[p]
        if kind == "trivial":
            return 0
        if kind == "barrier" or n <= 0:
            return self.start[p]
        if n >= self.total[p]:
            # n reaches past the last batch: (count - 1) batches hold fewer than total
            # elements, so ceil(n / unit) >= count and the time is the producer's end
            return self.end[p]
        key = (p, n)
        cached = self._ready_cache.get(key)
        if cached is not None:
            return cached
        m = self.model
        c_ub, span_ub = self.max_count[p], self.end_ub[p]
        k_raw = self._var(0, n, "kraw")
        m.add_division_equality(k_raw, n + self.unit[p] - 1, self.unit[p])
        k = self._var(1, max(c_ub, 1), "k")                 # n > 0: at least one batch
        m.add_min_equality(k, [k_raw, self.count[p]])
        j = self._var(0, max(c_ub - 1, 0), "j")             # batches after the first
        m.add(j == k - 1)
        d = self._var(1, max(c_ub - 1, 1), "d")             # gaps after the first batch, at least 1
        m.add_max_equality(d, [self.count[p] - 1, 1])
        span = self._var(0, span_ub, "span")
        m.add(span == self.end[p] - self.first[p])
        interval = self._var(0, span_ub, "interval")
        m.add_division_equality(interval, span, d)
        whole = self._var(0, span_ub, "whole")             # interval * d
        m.add_multiplication_equality(whole, [interval, d])
        rem = self._var(0, max(c_ub - 2, 0), "rem")
        m.add(rem == span - whole)
        j_interval = self._var(0, span_ub, "jinterval")
        m.add_multiplication_equality(j_interval, [j, interval])
        j_rem = self._var(0, max(c_ub - 1, 0) * max(c_ub - 2, 0), "jrem")
        m.add_multiplication_equality(j_rem, [j, rem])
        frac = self._var(0, c_ub, "frac")
        m.add_division_equality(frac, j_rem, d)
        t = self.first[p] + j_interval + frac
        self._ready_cache[key] = t
        return t

    def _latest(self, needs, ub: int, label: str):
        v = self._var(0, ub, label)
        self.model.add_max_equality(v, [self.ready(p, n) for p, n in needs])
        return v

    # -- nodes ---------------------------------------------------------------------

    def _build_node(self, i: int) -> None:
        kind, arg = self.mapper.node_form(i)
        self.kind[i] = kind
        if kind == "trivial":
            self.start[i] = self.end[i] = 0
        elif kind == "alias":
            r = self.root[arg]
            self.root[i] = r
            self.start[i], self.end[i], self.end_ub[i] = self.start[r], self.end[r], self.end_ub[r]
        elif kind == "barrier":
            ub = self._needs_ub(arg)
            self.start[i] = self.end[i] = self._latest(arg, ub, "barrier")
            self.end_ub[i] = ub
        else:
            self._build_pipelined(i, arg)

    def _build_pipelined(self, i: int, plan) -> None:
        """_pipelined_spec as constraints. The recipe's needs, batch terms, problem
        count and outputs per problem must not depend on pf (only pf, the hardware
        interval and the setup may); every recipe in xdsl_timing satisfies this."""
        m = self.model
        pfs = self.pf_modes.get(i)
        if not pfs:
            raise ValueError(f"CpsatTiming: no candidate pf for pipelined node {i}")
        recipes = {pf: plan(pf) for pf in pfs}
        r0 = recipes[pfs[0]]
        for pf, r in recipes.items():
            if (r.first_needs, r.all_needs, r.batch_terms, r.num_problems, r.outputs_per_problem) != (
                    r0.first_needs, r0.all_needs, r0.batch_terms, r0.num_problems, r0.outputs_per_problem):
                raise NotImplementedError(f"CpsatTiming: node {i}'s recipe changes shape with pf ({pf})")
        if len(pfs) > 1:
            r = self.tie.get(i, i)
            if r != i:
                if self.pf_modes.get(r) != pfs or r not in self.mode:
                    raise ValueError(f"CpsatTiming: node {i} is tied to node {r}, which has other candidates")
                self.mode[i] = self.mode[r]
            else:
                self.mode[i] = {pf: m.new_bool_var(f"pf{pf}_n{i}") for pf in pfs}
                m.add_exactly_one(self.mode[i].values())

        num_problems, opp = r0.num_problems, r0.outputs_per_problem
        self.total[i] = num_problems * opp
        self.max_count[i] = num_problems

        circuit_ticks = {pf: self._ticks(math.ceil(num_problems / r.pf) * r.hardware_interval)
                         for pf, r in recipes.items()}
        round_ticks = {pf: self._ticks(r.hardware_interval) for pf, r in recipes.items()}
        setup_ticks = {pf: self._ticks(r.setup) for pf, r in recipes.items()}
        start_ub = self._needs_ub(r0.first_needs) + max(setup_ticks.values())
        all_ub = max(self._needs_ub(r0.all_needs), start_ub)
        end_ub = max(start_ub + max(circuit_ticks.values()), all_ub + max(round_ticks.values()))
        self.end_ub[i] = end_ub

        # the op starts setup after its first problem's inputs; no round starts before it
        first_ready = self._latest(r0.first_needs, self._needs_ub(r0.first_needs), "firstready")
        start = self._var(0, start_ub, "start")
        m.add(start == first_ready + self.by_mode(i, setup_ticks))
        inputs_ready = self._latest(r0.all_needs, self._needs_ub(r0.all_needs), "allready")
        all_ready = self._var(0, all_ub, "roundready")
        m.add_max_equality(all_ready, [inputs_ready, start])
        circuit_end = start + self.by_mode(i, circuit_ticks)
        supply_end = all_ready + self.by_mode(i, round_ticks)
        end = self._var(0, end_ub, "end")
        m.add_max_equality(end, [circuit_end, supply_end])

        # circuit-limited (circuit_end >= supply_end): pf problems per batch; otherwise
        # the producers' problems per batch, at least one
        circuit_limited = m.new_bool_var(f"circuit_n{i}")
        m.add(circuit_end >= supply_end).only_enforce_if(circuit_limited)
        m.add(circuit_end <= supply_end - 1).only_enforce_if(circuit_limited.Not())
        # (when supply-limited, at most pf: a round finishes at most pf problems)
        per_round = self._var(1, num_problems, "perround")
        m.add(per_round == self.by_mode(i, {pf: min(r.pf, num_problems) for pf, r in recipes.items()}))
        per_batch = self._var(1, num_problems, "perbatch")
        m.add(per_batch == per_round).only_enforce_if(circuit_limited)
        supply_batch = self._var(1, num_problems, "supplyround")
        m.add_min_equality(supply_batch, [self._supply_per_batch(r0), per_round])
        m.add(per_batch == supply_batch).only_enforce_if(circuit_limited.Not())

        count = self._var(1, num_problems, "count")
        m.add_division_equality(count, num_problems + per_batch - 1, per_batch)
        unit = self._var(opp, num_problems * opp, "unit")
        m.add(unit == per_batch * opp)

        # the first batch: one round after the start (_pipelined_spec)
        self.first[i] = start + self.by_mode(i, round_ticks)
        self.start[i], self.end[i], self.unit[i], self.count[i] = start, end, unit, count

    def _supply_per_batch(self, recipe):
        """min(num_problems, max(1, floor(smallest batch term))); a producer with an
        unbounded batch (trivial, barrier) is never the smallest."""
        m = self.model
        terms, bound = [], 0
        for p, mul, div, post in recipe.batch_terms:
            p = self.root[p]
            if self.kind[p] != "pipelined":
                continue
            factor = mul * post
            ub = self.total[p] * factor // div
            b = self._var(0, ub, "batch")
            m.add_division_equality(b, self.unit[p] * factor, div)
            terms.append(b)
            bound = max(bound, ub)
        if not terms:
            # every producer is unbounded: the op is always circuit-limited
            return recipe.num_problems
        smallest = self._var(0, bound, "minbatch")
        m.add_min_equality(smallest, terms)
        at_least_one = self._var(1, max(bound, 1), "atleast1")
        m.add_max_equality(at_least_one, [1, smallest])
        v = self._var(1, recipe.num_problems, "supplybatch")
        m.add_min_equality(v, [recipe.num_problems, at_least_one])
        return v


def pf_ladder(pf_cap: int) -> list[int]:
    """The pfs greedy's search can reach (xdsl_search._next_pf): 1, doubling, capped."""
    pfs, pf = [1], 1
    while pf < pf_cap:
        pf = min(pf_cap, pf * 2)
        pfs.append(pf)
    return pfs


@dataclass
class SegmentSolve:
    """CP-SAT's own numbers for one segment."""
    status: str
    objective_ns: float           # the segment's end (its latest output), standalone
    bound_ns: float               # CP-SAT's lower bound on it
    solve_wall_s: float


@dataclass
class CpsatSearchResult:
    """A CP-SAT search's mapping (replayed through the real mapper, after greedy's
    lowering of the non-critical ops) and the solver's numbers per segment."""
    result: SearchResult
    segments: list[SegmentSolve]

    @property
    def status(self) -> str:
        return "OPTIMAL" if all(s.status == "OPTIMAL" for s in self.segments) else "FEASIBLE"

    @property
    def solve_wall_s(self) -> float:
        return sum(s.solve_wall_s for s in self.segments)


def _solve_segment(target, segment: Segment, stationary_cap: int, stationary_total: int, cap: int,
                   time_limit_s: float, workers: int, scale: int, verbose: bool,
                   correction_cap: int = 0, correction_total: int = 0) -> tuple[dict, SegmentSolve]:
    """The pf of every array op of one segment, solved on the segment alone (its
    standalone block: inputs ready at time 0). Returns ({region value: pf}, numbers).

      timing      CpsatTiming of the segment's block; every op other than the array ops
                  at the pf greedy gives it before its search (_maxout_noncritical_ops)
      stationary  the segment's resident tiles, each counted as its own array, at most
                  stationary_cap. The real packing never opens more arrays than tiles.
      volatile    the peak of the array ops' volatile arrays, each held over its op's
                  [start less its setup, end] widened by a margin for the integer
                  rounding (xdsl_hardware._peak_volatile_array_count):
                    single module  at most cap (HardwareArrayCounts: max(stationary,
                                   volatile) <= cap), with each staged dynamic rhs
                                   (HardwareMapping.held_rhs) over [the rhs's start,
                                   its matvec's start]
                    dual module    a peak per module (xdsl_dual_module_timing.op_module),
                                   analog A >= stationary_total (the whole block's resident
                                   copies share the analog arrays), digital D, A + D <= cap
                                   (DualModuleArrayCounts.total_array_count); each stored
                                   dynamic rhs (held_rhs) in the digital module over [the
                                   rhs's start, its matvec's end]
                    ensemble       a peak per pool (macros_at_pf), RRAM R >= stationary_total,
                                   SRAM V, R + correction_total + V <= cap
                                   (EnsembleArrayCounts.total_array_count); each staged
                                   dynamic rhs (held_rhs) in the SRAM over [the rhs's
                                   start, its matvec's start]
      correction  ensemble only: the segment's SRAM correction macros (held for the whole
                  run) at most correction_cap
      objective   the segment's latest output end"""
    dual = isinstance(target, DualModuleTarget)
    ensemble = isinstance(target, EnsembleAppTarget)
    mapper, hardware = initial_mapping(target, segment.block)
    _maxout_noncritical_ops(mapper, {})
    n = len(mapper.node_order)
    search = {i for i, v in enumerate(mapper.node_order) if type(_defining_op(v)) in MAPPED_OPERATION_OPS}
    pf_modes = {}
    for i in range(n):
        if mapper.node_form(i)[0] != "pipelined":
            continue
        pf_modes[i] = pf_ladder(mapper.pf_cap_at(i)) if i in search else [mapper.mapped_pf[i]]
    timing = CpsatTiming(mapper, pf_modes, scale=scale)
    m = timing.model

    stationary = []
    for stat_value, consumer_op in hardware.stationary_pairs():
        ci = mapper.node_index[consumer_op.results[0]]
        per_copy = hardware.stationary_arrays_per_copy(stat_value)
        stationary.append(timing.by_mode(ci, {pf: per_copy * hardware.stationary_copies_at_pf(pf)
                                              for pf in pf_modes[ci]}))
    if stationary:
        m.add(sum(stationary) <= stationary_cap)

    if ensemble:
        # SRAM correction macros, held for the whole run: one copy at most its segment cap
        correction = [timing.by_mode(i, {pf: hardware.macros_at_pf(mapper.node_order[i].owner, pf)[2]
                                         for pf in pf_modes[i]}) for i in sorted(search)]
        m.add(sum(correction) <= correction_cap)

    margin = 2 * n  # time units: well above the measured rounding (validate_against_mapper)
    pools = ("rram", "sram") if ensemble else ("analog", "digital") if dual else ("all",)
    intervals = {key: [] for key in pools}
    demands = {key: [] for key in pools}
    for i in sorted(search):
        op = mapper.node_order[i].owner
        # live from the start of its setup (OperationMapping.startup)
        recipe = mapper.node_form(i)[1]
        setups = {pf: timing._ticks(recipe(pf).setup) for pf in pf_modes[i]}
        if len(set(setups.values())) == 1:
            live_start = timing.start[i] - next(iter(setups.values())) - margin
        else:  # an interval's start must be affine
            live_start = timing._var(-margin - max(setups.values()), timing.end_ub[i], "livestart")
            m.add(live_start == timing.start[i] - timing.by_mode(i, setups) - margin)
        size = timing._var(0, timing.end_ub[i] + max(setups.values()) + 2 * margin + 1, "live")
        m.add(size == timing.end[i] + margin + 1 - live_start)
        lo, hi = live_start, timing.end[i] + margin + 1
        for pf in pf_modes[i]:
            if ensemble:
                rram, sram, _ = hardware.macros_at_pf(op, pf)
                per_pool = {"rram": rram, "sram": sram}
            else:
                per_pool = {op_module(op) if dual else "all": hardware.volatile_arrays_at_pf(op, pf)}
            for key, count in per_pool.items():
                if count <= 0:
                    continue
                if len(pf_modes[i]) == 1:
                    intervals[key].append(m.new_interval_var(lo, size, hi, f"live_n{i}_{key}"))
                else:
                    intervals[key].append(m.new_optional_interval_var(lo, size, hi, timing.mode[i][pf],
                                                                      f"live_n{i}_{key}_pf{pf}"))
                demands[key].append(count)
    # a held dynamic rhs (xdsl_hardware.OperationMapping.rhs_hold_arrays), at any pf, in its
    # matvec's pool: from the rhs's first batch to the matvec's start (1F1C and the
    # ensemble: staged, then copied by the setup) or end (dual module: the rhs's only copy)
    until = timing.start if hardware.HELD_RHS_UNTIL == "start" else timing.end
    for i, rhs_i, count in hardware.held_rhs():
        key = "sram" if ensemble else op_module(mapper.node_order[i].owner) if dual else "all"
        size = timing._var(0, timing.end_ub[i] + 2 * margin + 1, "held")
        m.add(size == until[i] - timing.start[rhs_i] + 2 * margin + 1)
        intervals[key].append(m.new_interval_var(timing.start[rhs_i] - margin, size,
                                                 until[i] + margin + 1, f"held_n{i}"))
        demands[key].append(count)
    if dual or ensemble:
        # one pool holds the resident copies (analog / RRAM): at least stationary_total;
        # an ensemble's SRAM also holds every op's correction for the whole run
        held, other = pools
        fixed = correction_total if ensemble else 0
        if stationary_total + fixed > cap:
            raise RuntimeError(f"the stationary arrays ({stationary_total} + {fixed}) exceed cap {cap}")
        held_cap = timing._var(stationary_total, cap, f"{held}_arrays")
        other_cap = timing._var(0, cap, f"{other}_arrays")
        m.add(held_cap + fixed + other_cap <= cap)
        m.add_cumulative(intervals[held], demands[held], held_cap)
        m.add_cumulative(intervals[other], demands[other], other_cap)
    else:
        m.add_cumulative(intervals["all"], demands["all"], cap)

    outputs = [mapper.node_index[v] for op in segment.block.ops if _is_return(op) for v in op.operands]
    finish = timing._var(0, max(timing.end_ub[i] for i in outputs), "finish")
    m.add_max_equality(finish, [timing.end[i] for i in outputs])
    m.minimize(finish)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    solver.parameters.log_search_progress = verbose
    status = solver.solve(m)
    status_name = solver.status_name(status)
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        raise RuntimeError(f"CP-SAT found no mapping for a {len(segment.region_values)}-op segment "
                           f"under cap {cap} ({status_name})")
    pfs = {}
    for i in search:
        modes = pf_modes[i]
        pfs[mapper.node_order[i]] = modes[0] if len(modes) == 1 else next(
            pf for pf in modes if solver.value(timing.mode[i][pf]))
    return pfs, SegmentSolve(status=status_name, objective_ns=solver.objective_value / scale,
                             bound_ns=solver.best_objective_bound / scale, solve_wall_s=solver.wall_time)


def solve_capped(target, block, cap: int, segments: list[Segment], stationary_caps: list[SegmentStationary],
                 time_limit_s: float, workers: int, scale: int = 1000, verbose: bool = False,
                 correction_caps: list[SegmentStationary] | None = None) -> CpsatSearchResult:
    """The pf of every array op (matvec, lut_search, lut_search_vector) under the cap,
    every other op as greedy treats it (maxed out before, lowered after:
    xdsl_search._maxout_noncritical_ops, _lower_noncritical_pf_to_neighbor_rate).

    Each distinct segment is solved once on its own (_solve_segment, up to time_limit_s
    each) and its pfs are copied onto every occurrence. The full block is then replayed
    with greedy's lowering. Each segment's resident tiles stay within its cap (at most
    greedy's tiles there) and, on the ensemble, its SRAM correction within correction_caps;
    the packed resident count and the volatile peak (occurrences that overlap in time add
    up) are only known after the replay, so a replayed mapping over the cap raises."""
    ensemble = isinstance(target, EnsembleAppTarget)
    if ensemble and correction_caps is None:
        raise ValueError("solve_capped: the ensemble needs correction_caps (sram_correction_by_segment)")
    full_pf = {}
    solves = []
    # the full block's packed resident arrays (the pool that holds them is at least this);
    # each segment's own cap is in tiles (stationary_by_segment)
    stationary_total = sum(s.total for s in stationary_caps)
    correction_total = (sum(c.cap_per_copy * seg.repeats for c, seg in zip(correction_caps, segments))
                        if ensemble else 0)
    for s, (seg, stationary) in enumerate(zip(segments, stationary_caps)):
        pfs, solve = _solve_segment(target, seg, stationary.cap_per_copy, stationary_total, cap, time_limit_s,
                                    workers, scale, verbose,
                                    correction_cap=correction_caps[s].cap_per_copy if ensemble else 0,
                                    correction_total=correction_total)
        solves.append(solve)
        for occ in seg.occurrences:
            for region_v, full_v in zip(seg.region_values, occ):
                if region_v in pfs:
                    full_pf[full_v] = pfs[region_v]

    mapper, hardware = initial_mapping(target, block)
    _maxout_noncritical_ops(mapper, {})
    mapper.pf_overrides.update({mapper.node_index[v]: pf for v, pf in full_pf.items()})
    mapper.compute_all()
    sink = mapper.node_order[sink_index(mapper)]
    if ensemble:
        budget = EnsembleBudget(macro_cap=cap, accel_area_caps={})
    elif isinstance(target, DualModuleTarget):
        budget = DualModuleBudget(array_cap=cap, accel_area_caps={})
    else:
        budget = Budget(nonvolatile_cap=cap, volatile_cap=cap, accel_area_caps={})
    mapper.set_memo(True)
    _lower_noncritical_pf_to_neighbor_rate(mapper, hardware, SearchMemo(), sink, budget,
                                           accel_area_index(block, mapper.node_index))
    mapper.compute_all()
    counts = rebuild_hardware(hardware)
    mapper.set_memo(False)
    over = _arrays_over_budget(counts, budget)
    if over is not None:
        raise RuntimeError(f"the replayed mapping has {over}")
    return CpsatSearchResult(result=_result(mapper, hardware, sink, counts), segments=solves)


def sink_index(mapper: TimingOnlyMapping) -> int:
    """Node position of the block's returned value."""
    return mapper.node_index[next(op for op in mapper.block.ops if _is_return(op)).operands[0]]


def validate_against_mapper(mapper: TimingOnlyMapping, scale: int = 1000, time_limit_s: float = 300.0,
                            workers: int = 8) -> dict:
    """Fixes every node at the pf the mapper last computed (mapper.mapped_pf, after
    compute_all) and compares CP-SAT's start / end of every node with mapper.specs.
    Returns the solve status and the largest absolute (ns) and relative differences."""
    n = len(mapper.node_order)
    pf_modes = {i: [mapper.mapped_pf[i]] for i in range(n) if mapper.node_form(i)[0] == "pipelined"}
    timing = CpsatTiming(mapper, pf_modes, scale=scale)
    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_s
    solver.parameters.num_workers = workers
    status = solver.solve(timing.model)
    out = {"status": solver.status_name(status), "nodes": n, "wall_s": round(solver.wall_time, 2)}
    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        out["invalid"] = timing.model.validate()[:300]
        return out
    worst_abs, worst_rel, worst_node = 0.0, 0.0, None
    for i in range(n):
        spec = mapper.specs[i]
        for field, expr in (("start", timing.start[i]), ("end", timing.end[i])):
            got = solver.value(expr) / scale
            want = getattr(spec, field)
            diff = abs(got - want)
            rel = diff / want if want > 0 else diff
            worst_abs = max(worst_abs, diff)
            if rel > worst_rel:
                worst_rel, worst_node = rel, (i, field, want, got)
    sink = sink_index(mapper)
    out.update({"max_abs_ns": round(worst_abs, 4), "max_rel": worst_rel, "worst": worst_node,
                "sink_end_ns": mapper.specs[sink].end, "sink_end_cpsat_ns": solver.value(timing.end[sink]) / scale})
    return out
