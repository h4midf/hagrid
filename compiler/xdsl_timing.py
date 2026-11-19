"""Timing model used by the search: TimingOnlyMapping.

For every value of a block it computes an OutputGenSpec -- when the value's output
starts to appear, in what batch size and at what rate -- given a parallelism factor
(pf) per op. The search only changes pf values in plain dicts; no IR is created or
changed here.
"""
import math

from xdsl.dialects.func import ReturnOp
from xdsl.ir import BlockArgument, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import CommonReshapeOp, CommonSliceOp, ConstRefOp
from compiler.dialects.rewrite_helpers import shape_of
from compiler.xdsl_fixed_point import bit_width
from custom_op_library.common.tools import load_results
from targets.target_loader import SingleModuleTarget
from targets.target_support import TargetTiming

# per-token int8 quantisation (custom_op_library/token_quant): the abs-max tree's inputs
# (its one characterized size; a wider row takes several passes) and the mantissa width
# of token_scale and quant_mant_round
ABSMAX_TREE_INPUTS = load_results.characterized_sizes("token_quant")[0]
TOKEN_QUANT_MANTISSA_WIDTH = 12

# shape ops: no cost of their own, they pass their operand's spec on
SHAPE_OPS = (cd.CimTransposeOp, cd.CimBroadcastOp, cd.CimViewOp, cd.CimPermuteOp, cd.CimReshapeOp,
             cd.CimUnfoldOp, CommonSliceOp, CommonReshapeOp)

# ops whose per-element work runs on shared 1-lane units (shared_lane_plan): their pf
# counts lanes
LANE_OPS = (cd.CimVecAddOp, cd.CimVecElMulOp, cd.CimVecSelectOp, cd.CimSoftmaxOp, cd.CimQuantizePerTokenOp)


def lane_costs(op, t) -> tuple:
    """(rows, row width, round ns, per-element stage delays ns) of a LANE_OPS op on target
    timing t. A row is a vector (vec_add / vec_el_mul / vec_select) or a softmax or
    quantize_per_token row. Its round: a WMU read, the row's circuit (one 1-lane unit
    delay for a vector op; the row pipeline for softmax, its per-element stages counted
    one element delay each; token_scale + one rounding delay for quantize_per_token),
    a WMU write."""
    width = shape_of(op.results[0].type)[-1]
    rows = math.prod(shape_of(op.results[0].type)) // width
    w = _accel_width_of(op, "accel_data_width") if type(op) is not cd.CimSoftmaxOp else None
    if type(op) is cd.CimVecAddOp:
        circuit, delays = t.vecadd_accel(1, w), (load_results.vecadd_accel_ns(1, w),)
    elif type(op) is cd.CimVecElMulOp:
        circuit, delays = t.vecmul_accel(1, w), (load_results.vecmul_accel_ns(1, w),)
    elif type(op) is cd.CimVecSelectOp:
        circuit, delays = t.vecselect_accel(1, w), (load_results.vecselect_accel_ns(1, w),)
    elif type(op) is cd.CimSoftmaxOp:
        variant = op.variant.data
        circuit = t.softmax_row_pipeline_ns(variant, width)
        delays = tuple(row["delay_ps"] / 1000.0 for _, _, row, per_element, _ in
                       load_results.softmax_stages(variant, width) if per_element)
    elif type(op) is cd.CimQuantizePerTokenOp:
        mw = TOKEN_QUANT_MANTISSA_WIDTH
        circuit = t.token_scale(w, mw) + t.quantize_cmos("mant_round", w, mw)
        delays = (load_results.quantize_accel_ns("mant_round", w, mw),)
    else:
        raise TypeError(f"{op.name} is not one of LANE_OPS")
    return rows, width, t.wmu_read() + circuit + t.wmu_write(), delays


def _lanes_per_row(width: int, round_ns: float, delays: tuple) -> list[int]:
    """Lanes of each per-element stage that finish one row's elements in one round."""
    return [max(1, math.ceil(width * d / round_ns)) for d in delays]


def lane_pf_cap(rows: int, width: int, round_ns: float, delays: tuple) -> int:
    """pf (lanes) with every row in flight at once."""
    return rows * max(_lanes_per_row(width, round_ns, delays))


def shared_lane_plan(pf: int, rows: int, width: int, round_ns: float, delays: tuple) -> tuple:
    """(rows in flight, round ns, lanes of each per-element stage) at pf lanes. Each lane
    is busy one stage delay per element, so lanes are shared over the round. pf counts
    the lanes of the stage needing the most (full = its lanes for one row per round):
      pf >= full   pf // full rows in flight, each one round; the stages' lanes for
                   those rows
      pf <  full   one row at a time; each stage gets its share pf / full of its
                   one-row lanes (at least one), and the round stretches to the slowest
                   stage's ceil(width / lanes) element delays when that is longer"""
    per_row = _lanes_per_row(width, round_ns, delays)
    full = max(per_row)
    if pf >= full:
        in_flight = min(rows, pf // full)
        return in_flight, round_ns, [math.ceil(in_flight * width * d / round_ns) for d in delays]
    lanes = [max(1, math.ceil(p * pf / full)) for p in per_row]
    return 1, max(round_ns, max(math.ceil(width / n) * d for n, d in zip(lanes, delays))), lanes


class OutputGenSpec:
    """How an op's output becomes available: the op starts at `start`, its first batch
    of `unit_size` elements is out at `first`, the other `count` - 1 batches follow
    evenly up to the last at `end`. interval = (end - start) / count, the mean time per
    batch (reported, not used for readiness).
    circuit_bound: the op's circuits, not its inputs, set `end` (it falls behind its
    inputs); False for an op that is not pipelined.
    """
    __slots__ = ("start", "first", "interval", "unit_size", "count", "end", "circuit_bound")

    def __init__(self, start: float, first: float, interval: float, unit_size: float, count: int, end: float,
                 circuit_bound: bool):
        self.start = start
        self.first = first
        self.interval = interval
        self.unit_size = unit_size
        self.count = count
        self.end = end
        self.circuit_bound = circuit_bound

    def __repr__(self) -> str:
        return (f"OutputGenSpec(start={self.start}, first={self.first}, interval={self.interval}, "
                f"unit_size={self.unit_size}, count={self.count}, end={self.end}, "
                f"circuit_bound={self.circuit_bound})")


# spec of a value ready at time 0 (arguments, constants); one shared object: specs are never mutated
_TRIVIAL_SPEC = OutputGenSpec(start=0.0, first=0.0, interval=0.0, unit_size=float("inf"), count=1, end=0.0,
                              circuit_bound=False)


def _time_for_n_elements(spec: OutputGenSpec, n) -> float:
    """Time at which the first n elements of a value are available: the k-th batch
    (k = ceil(n / unit_size)) at first + (k - 1) * (end - first) / (count - 1)."""
    if math.isinf(spec.unit_size):
        return spec.start
    k = min(math.ceil(n / spec.unit_size), spec.count)
    if k <= 1:
        return spec.first
    return spec.first + (k - 1) * (spec.end - spec.first) / (spec.count - 1)


def _pipelined_spec(
    first_ready, all_ready, num_problems, problems_per_input_batch, outputs_per_problem,
    hardware_interval, pf, setup,
) -> OutputGenSpec:
    """pf circuits, each starting the next problem as soon as that problem's inputs
    have arrived. The op starts setup after the first problem's inputs are ready
    (first_ready; setup: work before the first round, e.g. writing an operand's copies)
    and ends when the slower of two limits is done: the circuits (ceil(num_problems / pf)
    rounds of hardware_interval) or the inputs (the last arrives at all_ready, plus one
    round, and no round starts before the op does). Output batches follow the limit: pf
    problems per round when the circuits limit; when the inputs do, the producer's
    problems per batch, at most pf (a round finishes at most pf problems). Either way a
    batch is at most one round's problems, and the first batch's inputs are there at
    first_ready (they come in the producer's first batch), so it is out one round after
    the start; the rest follow evenly up to the end.
    """
    start = first_ready + setup
    circuit_end = start + math.ceil(num_problems / pf) * hardware_interval
    supply_end = max(all_ready, start) + hardware_interval
    circuit_bound = circuit_end > supply_end
    if circuit_end >= supply_end:
        end = circuit_end
        problems_per_batch = min(pf, num_problems)
    else:
        end = supply_end
        # at least one problem per output batch, even if a producer batch completes less
        problems_per_batch = min(num_problems, pf, max(1, int(problems_per_input_batch)))
    count = math.ceil(num_problems / problems_per_batch)
    # never after end: both circuit_end and supply_end are at least start + one round
    first = start + hardware_interval
    return OutputGenSpec(start=start, first=first, interval=(end - start) / count,
                         unit_size=problems_per_batch * outputs_per_problem, count=count, end=end,
                         circuit_bound=circuit_bound)


class PipelineRecipe:
    """The inputs of one _pipelined_spec call at one pf, as data: everything except the
    operands' specs, which are referenced by node position. evaluate_recipe turns it
    into the spec; other models of the same timing (a solver) read the same fields.

    first_needs / all_needs  ((operand position, n), ...): the first / last problem can
                             start once each listed operand has its first n elements;
                             first_ready / all_ready are the latest of these times.
    batch_terms              ((operand position, mul, div, post_mul), ...): the producers'
                             problems per input batch are the smallest operand
                             unit_size * mul / div * post_mul (in that order).
    setup                    ns before the first round, once the first problem's inputs
                             are ready (0 for an op that starts right away); like
                             hardware_interval and pf, it may change with pf.
    """
    __slots__ = ("first_needs", "all_needs", "num_problems", "batch_terms", "outputs_per_problem",
                 "hardware_interval", "pf", "setup")

    def __init__(self, first_needs: tuple, all_needs: tuple, num_problems: int, batch_terms: tuple,
                 outputs_per_problem: int, hardware_interval: float, pf: int, setup: float):
        if setup < 0:
            raise ValueError(f"PipelineRecipe: negative setup {setup}")
        self.first_needs = first_needs
        self.all_needs = all_needs
        self.num_problems = num_problems
        self.batch_terms = batch_terms
        self.outputs_per_problem = outputs_per_problem
        self.hardware_interval = hardware_interval
        self.pf = pf
        self.setup = setup

    def __repr__(self) -> str:
        return (f"PipelineRecipe(first_needs={self.first_needs}, all_needs={self.all_needs}, "
                f"num_problems={self.num_problems}, batch_terms={self.batch_terms}, "
                f"outputs_per_problem={self.outputs_per_problem}, "
                f"hardware_interval={self.hardware_interval}, pf={self.pf}, setup={self.setup})")


def _latest_ready(specs, needs: tuple) -> float:
    """When every (operand position, n) of `needs` has its first n elements."""
    t = None
    for idx, n in needs:
        ready = _time_for_n_elements(specs[idx], n)
        if t is None or ready > t:
            t = ready
    return t


def evaluate_recipe(recipe: PipelineRecipe, specs) -> OutputGenSpec:
    """The spec of a recipe given its operands' specs (by node position)."""
    batch = None
    for idx, mul, div, post_mul in recipe.batch_terms:
        b = specs[idx].unit_size * mul / div * post_mul
        if batch is None or b < batch:
            batch = b
    return _pipelined_spec(
        first_ready=_latest_ready(specs, recipe.first_needs),
        all_ready=_latest_ready(specs, recipe.all_needs),
        num_problems=recipe.num_problems, problems_per_input_batch=batch,
        outputs_per_problem=recipe.outputs_per_problem,
        hardware_interval=recipe.hardware_interval, pf=recipe.pf, setup=recipe.setup,
    )


def matvec_recipe(
    lhs_i, lhs_numel, rhs_i, rhs_numel,
    row_size, total_rows, output_width,
    hardware_interval, pf, setup,
) -> PipelineRecipe:
    """Matvec: pf copies, each computing one input row (row_size inputs, output_width
    outputs) per round, once the whole rhs is ready and setup has passed (e.g. the rhs
    copies written, xdsl_ensemble_timing).
    """
    return PipelineRecipe(
        first_needs=((rhs_i, rhs_numel), (lhs_i, row_size)),
        all_needs=((rhs_i, rhs_numel), (lhs_i, lhs_numel)),
        num_problems=total_rows, batch_terms=((lhs_i, 1, row_size, 1),),
        outputs_per_problem=output_width, hardware_interval=hardware_interval, pf=pf, setup=setup,
    )


def _broadcast_factor(total_elements: int, operand_numel: int) -> int:
    """How many output elements share one element of an operand: >1 for an operand
    broadcast over the output (e.g. one value per row), 1 otherwise.
    """
    return max(1, total_elements // operand_numel)


def vector_op_recipe(
    a_i, a_numel, b_i, b_numel,
    vector_size, total_elements, hardware_interval, pf,
) -> PipelineRecipe:
    """Two-operand elementwise op (vec_add, vec_el_mul, scatter): pf circuits, each
    computing one vector of vector_size outputs per round. A broadcast operand is
    needed proportionally less (_broadcast_factor).
    """
    a_factor = _broadcast_factor(total_elements, a_numel)
    b_factor = _broadcast_factor(total_elements, b_numel)
    return PipelineRecipe(
        first_needs=((a_i, math.ceil(vector_size / a_factor)), (b_i, math.ceil(vector_size / b_factor))),
        all_needs=((a_i, a_numel), (b_i, b_numel)),
        num_problems=math.ceil(total_elements / vector_size),
        batch_terms=((a_i, a_factor, vector_size, 1), (b_i, b_factor, vector_size, 1)),
        outputs_per_problem=vector_size, hardware_interval=hardware_interval, pf=pf, setup=0.0,
    )


def lut_search_recipe(index_i, total_lookups, output_width, hardware_interval, pf) -> PipelineRecipe:
    """One-operand op with one evaluation per input element (lut_search,
    lut_search_vector and the pointwise accelerators): pf circuits, one element each
    per round, output_width outputs per element.
    """
    return PipelineRecipe(
        first_needs=((index_i, 1),), all_needs=((index_i, total_lookups),),
        num_problems=total_lookups, batch_terms=((index_i, 1, 1, 1),),
        outputs_per_problem=output_width, hardware_interval=hardware_interval, pf=pf, setup=0.0,
    )


def reduce_axis_recipe(
    parent_i, reduce_size, inner, num_problems, outputs_per_problem, hardware_interval, pf,
) -> PipelineRecipe:
    """Reduction over one axis of an input [outer, reduce_size, inner] that arrives in
    row-major order: pf circuits, one problem each per round; a problem reduces
    reduce_size inputs and gives outputs_per_problem outputs (k for top-k). Problem p
    reduces the inputs at (p // inner) * reduce_size * inner + r * inner + p % inner,
    r < reduce_size, so problem 0's last input is element (reduce_size - 1) * inner.
    """
    return PipelineRecipe(
        first_needs=((parent_i, (reduce_size - 1) * inner + 1),),
        all_needs=((parent_i, reduce_size * num_problems),),
        num_problems=num_problems, batch_terms=((parent_i, 1, reduce_size, 1),),
        outputs_per_problem=outputs_per_problem, hardware_interval=hardware_interval, pf=pf, setup=0.0,
    )


def row_op_recipe(parent_i, row_width, num_rows, hardware_interval, pf) -> PipelineRecipe:
    """Op over whole rows of row_width elements whose output has the input's size
    (softmax, layer_norm, rms_norm): pf circuits, one row each per round.
    """
    return PipelineRecipe(
        first_needs=((parent_i, row_width),), all_needs=((parent_i, num_rows * row_width),),
        num_problems=num_rows, batch_terms=((parent_i, 1, row_width, 1),),
        outputs_per_problem=row_width, hardware_interval=hardware_interval, pf=pf, setup=0.0,
    )


def reduce_vec_matvec_tiles_recipe(
    parent_i, reduce_size, output_width, num_rows, per_pass_cost, pf,
) -> PipelineRecipe:
    """Sum of a matvec's row-tile partial sums: each output row (output_width outputs)
    adds reduce_size tile vectors. pf adder lanes: min(pf, output_width) lanes per row
    and pf // output_width rows at once.
    """
    lanes = min(pf, output_width)
    rows_at_once = max(1, pf // output_width)
    inputs_per_row = reduce_size * output_width
    return PipelineRecipe(
        first_needs=((parent_i, inputs_per_row),), all_needs=((parent_i, inputs_per_row * num_rows),),
        num_problems=num_rows, batch_terms=((parent_i, 1, inputs_per_row, 1),),
        outputs_per_problem=output_width,
        hardware_interval=math.ceil(output_width / lanes) * per_pass_cost,
        pf=rows_at_once, setup=0.0,
    )


_MISSING = object()

# size bound of the spec memos (_spec_cache, _spec_intern); clearing one only costs recomputation
_SPEC_CACHE_MAX_ENTRIES = 3_000_000


class _SpecOverlay:
    """Dict-like view keyed by node position: overlay entries first, then the baseline."""

    def __init__(self, overlay: dict, baseline: dict):
        self._overlay = overlay
        self._baseline = baseline

    def __getitem__(self, idx: int) -> OutputGenSpec:
        found = self._overlay.get(idx, _MISSING)
        if found is not _MISSING:
            return found
        return self._baseline[idx]


def _stamped_attr(op, field: str):
    """Accelerator attribute stamped on the op by xdsl_accel_width_stamp.py. Missing
    means the stamping pass did not run.
    """
    attr = op.attributes.get(field)
    if attr is None:
        raise ValueError(
            f"TimingOnlyMapping: {type(op).__name__} is missing the {field!r} attribute -- "
            "compiler/xdsl_accel_width_stamp.py's stamp_accel_widths(module, config) must run "
            "before this op reaches TimingOnlyMapping"
        )
    return attr


def _accel_width_of(op, field: str) -> int:
    """Integer stamped attribute (reduce_width, argmax_width, accel_data_width).
    """
    return _stamped_attr(op, field).value.data


def _accel_impl_of(op, field: str) -> str:
    """String stamped attribute (reduce_impl)."""
    return _stamped_attr(op, field).data


def _set_bit_indices(mask: int, start: int, end: int):
    """Indices in [start, end] whose bit is set in mask, in increasing order."""
    bits = bin(mask >> start)[:1:-1]
    limit = end - start
    k = bits.find("1")
    while k != -1 and k <= limit:
        yield start + k
        k = bits.find("1", k + 1)


class TimingOnlyMapping:
    """Computes an OutputGenSpec for every value of a block, given a pf per op.

    Ops with a latency: matvec, reduce_vec, lut_search(_vector), vec_el_mul, vec_add,
    argmax, max, scatter, and the accelerator ops (relu, gelu, softmax, silu, elu,
    leaky_relu, layer_norm, rms_norm, quantize, dequantize, rsqrt). Shape ops (transpose,
    broadcast, view, permute, reshape, unfold, slice, embedding) pass their operand's
    spec through; cat is ready once all its inputs are; block arguments, constants and
    save_stationary are ready at 0. Anything else raises.

    Bound to one block at construction. Per-node state (pf_overrides, mapped_pf,
    specs) is keyed by the node's position in node_order, not by SSAValue: int
    hashing is far cheaper at the search's call volume.
    """

    def __init__(self, target: SingleModuleTarget, block):
        self.target = target
        self._init_timing(target)
        self.block = block
        # requested pf per node (may exceed the op's cap)
        self.pf_overrides: dict[int, int] = {}
        # pf actually used per node: pf_overrides clamped to [1, cap]
        self.mapped_pf: dict[int, int] = {}
        # node order: block arguments, then each op's result (fixed for the mapper's lifetime)
        self.node_order: list[SSAValue] = list(block.args) + [
            op.results[0] for op in block.ops if op.results and not _is_return(op)
        ]
        self.node_index: dict[SSAValue, int] = {v: i for i, v in enumerate(self.node_order)}
        # dependency bitmasks, built lazily (_build_forward_reach)
        self._forward_reach: list[int] | None = None
        # node position -> pf cap (pf_cap_at)
        self._pf_cap_cache: dict[int, int] = {}
        # node position -> dynamic parents, as values and as positions (dynamic_parents)
        self._dynamic_parents_cache: list[list[SSAValue] | None] = [None] * len(self.node_order)
        self._dynamic_parent_idx_cache: list[tuple | None] = [None] * len(self.node_order)
        self._numel_cache: dict[SSAValue, int] = {}
        self._shape_cache: dict[SSAValue, tuple] = {}
        self.specs: dict[int, OutputGenSpec] = {}
        # memo of _build_spec
        self._spec_cache: dict = {}
        # field-value tuple -> canonical OutputGenSpec (_intern_spec)
        self._spec_intern: dict = {}
        # node position -> recipe(pf) or None (_build_spec)
        self._spec_plan_cache: dict = {}
        # node position -> its operands' node positions
        self._operand_idx_cache: dict[int, tuple] = {}
        # the current pf state while memos are on (set_memo), None otherwise or while
        # specs may not match pf_overrides; keys propagate_and_measure's memo and the
        # search's memos
        self._memo_enabled = False
        self.memo_state: frozenset | None = None
        self._measure_memo: dict = {}
        # see reusable_overlay
        self._last_overlay_key = None
        self._last_overlay: dict | None = None
        self._last_overlay_end: int | None = None

    def _init_timing(self, target: SingleModuleTarget) -> None:
        """The op costs of a single-module target (RMU / WMU and accelerator timings)."""
        self.target_timing = TargetTiming(target)

    def _at(self, specs: dict, operand: SSAValue) -> OutputGenSpec:
        """Spec of an operand (looks up the operand's node position)."""
        return specs[self.node_index[operand]]

    def _shape(self, value: SSAValue) -> tuple:
        cached = self._shape_cache.get(value)
        if cached is not None:
            return cached
        result = tuple(shape_of(value.type))
        self._shape_cache[value] = result
        return result

    def _numel(self, value: SSAValue) -> int:
        cached = self._numel_cache.get(value)
        if cached is not None:
            return cached
        result = math.prod(self._shape(value))
        self._numel_cache[value] = result
        return result

    def _build_forward_reach(self) -> list[int]:
        """Per node, a bitmask: bit j is set iff node j is this node or depends on it
        (transitively). Lets incremental recomputation skip nodes that come later in
        topological order but do not depend on a change. Built once, in one
        reverse-topological pass.
        """
        n = len(self.node_order)
        reach = [0] * n
        for i in range(n - 1, -1, -1):
            value = self.node_order[i]
            bits = 1 << i
            for use in value.uses:
                op = use.operation
                if _is_return(op) or not op.results:
                    continue
                consumer = op.results[0]
                j = self.node_index.get(consumer)
                if j is not None:
                    bits |= reach[j]
            reach[i] = bits
        return reach

    def affected_indices(self, changed_values) -> list[int]:
        """Positions of the nodes that depend on `changed_values` (themselves included),
        in increasing order: the nodes compute_from rewrites."""
        if self._forward_reach is None:
            self._forward_reach = self._build_forward_reach()
        forward_reach = self._forward_reach
        affected = 0
        for v in changed_values:
            affected |= forward_reach[self.node_index[v]]
        start_idx = min(self.node_index[v] for v in changed_values)
        return list(_set_bit_indices(affected, start_idx, len(self.node_order) - 1))

    def compute_from(self, changed_values) -> None:
        """Recomputes the specs of the nodes that depend on `changed_values` only;
        every other node's spec cannot have changed. Needs compute_all to have run once.
        """
        specs = self.specs
        for i in self.affected_indices(changed_values):
            # a node without an override runs at pf 1
            current_pf = min(self.pf_overrides.get(i, 1), self.pf_cap_at(i))
            specs[i] = self._build_spec(i, specs, current_pf)
            self.mapped_pf[i] = current_pf
        self.memo_state = None

    def set_memo(self, enabled: bool) -> None:
        """Turns the memos keyed on memo_state on or off. While on, every compute_all
        sets memo_state to the pf state it computed; compute_from clears it."""
        self._memo_enabled = enabled
        self.memo_state = frozenset(self.pf_overrides.items()) if enabled else None

    def compute_all(self) -> None:
        """Specs of every node of the block at the current pf_overrides."""
        specs: dict[int, OutputGenSpec] = {}
        for i in range(len(self.node_order)):
            # a node without an override runs at pf 1
            current_pf = min(self.pf_overrides.get(i, 1), self.pf_cap_at(i))
            specs[i] = self._build_spec(i, specs, current_pf)
            self.mapped_pf[i] = current_pf
        self.specs = specs
        self.memo_state = frozenset(self.pf_overrides.items()) if self._memo_enabled else None

    def propagate_and_measure(self, pf_changes: dict, measured_at: SSAValue) -> float:
        """How much earlier `measured_at` finishes if `pf_changes` were applied, without
        changing the mapper's state. Memoized while memo_state is set.
        """
        state = self.memo_state
        if state is None or not pf_changes:
            return self._propagate_and_measure_uncached(pf_changes, measured_at)
        changes_key = frozenset(pf_changes.items())
        key = (state, changes_key, measured_at)
        cached = self._measure_memo.get(key)
        if cached is None:
            cached = self._measure_memo[key] = self._propagate_and_measure_uncached(pf_changes, measured_at)
            # the overlay just computed is what the next array-cost check needs (reusable_overlay)
            self._last_overlay_key = (state, changes_key)
        return cached

    def reusable_overlay(self, changes_key: frozenset, changes: dict):
        """The overlay the last uncached propagate_and_measure computed, if it is
        exactly what compute_from(changes) would write into specs: same state
        and changes, measured at the last node, and every changed pf already at most its
        pf_cap. Otherwise None.
        """
        state = self.memo_state
        if state is None or self._last_overlay_key != (state, changes_key):
            return None
        if self._last_overlay_end != len(self.node_order) - 1:
            return None
        for value, pf in changes.items():
            if pf > self.pf_cap_at(self.node_index[value]):
                return None
        return self._last_overlay

    def _propagate_and_measure_uncached(self, pf_changes: dict, measured_at: SSAValue) -> float:
        if not pf_changes:
            return 0.0
        start_idx = min(self.node_index[v] for v in pf_changes)
        end_idx = self.node_index[measured_at]
        if end_idx < start_idx:
            return 0.0

        overlay: dict[int, OutputGenSpec] = {}
        gain = self._propagate_into(overlay, pf_changes, pf_changes, start_idx, end_idx)
        self._last_overlay = overlay
        self._last_overlay_end = end_idx
        return gain

    def _propagate_into(self, overlay: dict, changed_nodes, pf_changes: dict, start_idx: int, end_idx: int) -> float:
        """Recomputes into `overlay` every node in [start_idx, end_idx] that depends on
        `changed_nodes`, with `pf_changes` applied; returns how much earlier the node at
        end_idx finishes than in specs.
        """
        if self._forward_reach is None:
            self._forward_reach = self._build_forward_reach()
        affected = 0
        for v in changed_nodes:
            affected |= self._forward_reach[self.node_index[v]]

        overlay_specs = _SpecOverlay(overlay, self.specs)
        pf_at = {self.node_index[v]: pf for v, pf in pf_changes.items()}
        for i in _set_bit_indices(affected, start_idx, end_idx):
            pf = pf_at[i] if i in pf_at else self.mapped_pf[i]
            overlay[i] = self._build_spec(i, overlay_specs, pf)

        baseline_end = self.specs[end_idx].end
        new_spec = overlay.get(end_idx)
        new_end = new_spec.end if new_spec is not None else baseline_end
        return baseline_end - new_end

    def propagate_and_measure_chained(
        self, pf_changes: dict, measured_at: SSAValue, base_overlay: dict, base_pf_changes: dict,
    ):
        """Like propagate_and_measure, starting from a previous candidate's overlay
        (base_overlay, base_pf_changes): only the changes that differ from it are
        recomputed. Returns (gain, overlay).
        """
        if not pf_changes:
            return 0.0, dict(base_overlay)

        changed_nodes = [v for v in pf_changes if base_pf_changes.get(v) != pf_changes[v]]
        end_idx = self.node_index[measured_at]

        if not changed_nodes:
            cached = base_overlay.get(end_idx)
            new_end = cached.end if cached is not None else self.specs[end_idx].end
            baseline_end = self.specs[end_idx].end
            return baseline_end - new_end, dict(base_overlay)

        start_idx = min(self.node_index[v] for v in changed_nodes)
        if end_idx < start_idx:
            raise RuntimeError("propagate_and_measure_chained: measured_at is not downstream of the changed values")

        overlay = dict(base_overlay)
        gain = self._propagate_into(overlay, changed_nodes, pf_changes, start_idx, end_idx)
        return gain, overlay

    def pf_cap_at(self, idx: int) -> int:
        """pf cap of the node at position idx (_pf_cap_for_value_uncached), memoized by position."""
        cap = self._pf_cap_cache.get(idx)
        if cap is None:
            cap = self._pf_cap_cache[idx] = self._pf_cap_for_value_uncached(self.node_order[idx])
        return cap

    def pf_cap_for_value(self, value: SSAValue) -> int:
        """pf cap of a value's node."""
        return self.pf_cap_at(self.node_index[value])

    def dynamic_parents(self, value: SSAValue) -> list[SSAValue]:
        """Operands of `value` produced by ops (not block arguments or save_stationary):
        the dependencies the search walks. A cat has none: otherwise every cat would look
        like a merge point to the search's sync points, and skip-connection cats (e.g. a
        U-Net's) would spread one sync group over almost the whole graph. Memoized."""
        idx = self.node_index[value]
        parents = self._dynamic_parents_cache[idx]
        if parents is None:
            parents = self._dynamic_parents_cache[idx] = self._dynamic_parents_uncached(value)
        return parents

    def dynamic_parent_indices(self, idx: int) -> tuple:
        """dynamic_parents of the node at idx, as node positions. Memoized."""
        parents = self._dynamic_parent_idx_cache[idx]
        if parents is None:
            parents = self._dynamic_parent_idx_cache[idx] = tuple(
                self.node_index[p] for p in self.dynamic_parents(self.node_order[idx])
            )
        return parents

    def _dynamic_parents_uncached(self, value: SSAValue) -> list[SSAValue]:
        op = _defining_op(value)
        if op is None or type(op) is cd.CimCatOp:
            return []
        parents = []
        for operand in op.operands:
            parent_op = _defining_op(operand)
            if parent_op is None or type(parent_op) is cd.CimSaveStationaryOp:
                continue
            parents.append(operand)
        return parents

    def _pf_cap_for_value_uncached(self, value: SSAValue) -> int:
        """Largest useful pf of an op (its number of independent problems): input rows
        for matvec; output elements for reduce_vec, lut_search and the pointwise
        accelerator ops; rows for softmax, layer_norm and rms_norm; lookups for
        lut_search_vector; lanes for vec_el_mul/vec_add/vec_select, enough for every
        vector at once (_vector_op_recipe); top-k problems for
        argmax/max; 1 for everything else. The extra stationary copies a large pf needs
        are charged by xdsl_hardware, not limited here.
        """
        op = _defining_op(value)
        if type(op) is cd.CimMatvecOp:
            lhs_shape = self._shape(op.lhs)
            row_dims = lhs_shape[:-2] if len(self._shape(op.rhs)) == 3 else lhs_shape[:-1]
            return math.prod(row_dims)
        if type(op) is cd.CimReduceVecOp:
            return self._numel(value)
        if type(op) is cd.CimLutSearchOp:
            return self._numel(value)
        if type(op) in (
            cd.CimReluOp, cd.CimGeluOp, cd.CimSiluOp, cd.CimEluOp, cd.CimLeakyReluOp,
            cd.CimQuantizeOp, cd.CimDequantizeOp, cd.CimRsqrtOp,
        ):
            return self._numel(value)
        if type(op) in LANE_OPS:
            # pf counts lanes: up to every row (vector) in flight (shared_lane_plan)
            return lane_pf_cap(*lane_costs(op, self.target_timing))
        if type(op) in (cd.CimLayerNormOp, cd.CimRMSNormOp):
            # pf counts rows of hidden_size
            return self._numel(value) // int_from_prop(op.hidden_size)
        if type(op) is cd.CimLutSearchVectorOp:
            # pf counts lookups; each returns a whole row
            return self._numel(op.rhs)
        if type(op) is cd.CimRowAbsmaxOp:
            # pf counts rows: the output has one value per row
            return self._numel(value)
        if type(op) in (cd.CimArgmaxOp, cd.CimMaxOp):
            # independent top-k problems: the output has k values per problem
            k = op.k.value.data if op.k is not None else 1
            return self._numel(value) // k
        return 1

    def spec_at_pf(self, idx: int, pf: int) -> OutputGenSpec:
        """The spec the node at position idx would have at this pf, its operands as now."""
        return self._build_spec(idx, self.specs, pf)

    def _build_spec(self, idx: int, specs: dict, pf: int):
        """Spec of the node at position idx for this pf. Memoized on (node, pf, operand
        spec objects): specs are immutable and interned (_intern_spec), so identical
        objects mean identical values.
        """
        value = self.node_order[idx]
        if type(value) is BlockArgument:
            return _TRIVIAL_SPEC
        op = value.owner
        if type(op) in (ConstRefOp, cd.CimSaveStationaryOp):
            return _TRIVIAL_SPEC

        # operand positions are fixed IR structure: resolved once per node
        operand_idx = self._operand_idx_cache.get(idx)
        if operand_idx is None:
            operand_idx = tuple(self.node_index[operand] for operand in op.operands)
            self._operand_idx_cache[idx] = operand_idx
        key = (idx, pf, tuple([specs[j] for j in operand_idx]))
        cached = self._spec_cache.get(key)
        if cached is not None:
            return cached
        plan = self._plan_at(idx, op)
        if plan is not None:
            spec = evaluate_recipe(plan(pf), specs)
        else:
            spec = self._build_spec_uncached(op, specs)
        if len(self._spec_cache) >= _SPEC_CACHE_MAX_ENTRIES:
            # bounded, see _SPEC_CACHE_MAX_ENTRIES (cleared in place: the dicts may be shared)
            self._spec_cache.clear()
            self._spec_intern.clear()
        result = self._intern_spec(spec)
        self._spec_cache[key] = result
        return result

    def _intern_spec(self, spec: OutputGenSpec) -> OutputGenSpec:
        """One canonical object per distinct field values, so _build_spec's
        identity-keyed cache hits for equal specs. Zero fields also key on their sign.
        """
        values = (spec.start, spec.first, spec.interval, spec.unit_size, spec.count, spec.end, spec.circuit_bound)
        if 0.0 in values:
            values = values + tuple(math.copysign(1.0, v) for v in values)
        canonical = self._spec_intern.get(values)
        if canonical is None:
            self._spec_intern[values] = spec
            return spec
        return canonical

    def _plan_at(self, idx: int, op):
        """The node's recipe(pf), or None for an op without a cost of its own. Ops in
        _SPEC_PLANNERS do their op-constant setup once per node (the plan); each later
        spec only builds the recipe for its pf."""
        plan = self._spec_plan_cache.get(idx, _MISSING)
        if plan is _MISSING:
            planner = self._SPEC_PLANNERS.get(type(op))
            plan = self._spec_plan_cache[idx] = planner(self, op) if planner is not None else None
        return plan

    def setup_at(self, idx: int) -> float:
        """The setup (ns) of the node at position idx at its mapped pf (PipelineRecipe):
        0 for a node that is not pipelined."""
        kind, recipe = self.node_form(idx)
        return recipe(self.mapped_pf[idx]).setup if kind == "pipelined" else 0.0

    def node_form(self, idx: int):
        """How the spec of the node at position idx is formed, for models of this timing
        other than _build_spec (a solver):
          ("trivial", None)          ready at 0 (_TRIVIAL_SPEC)
          ("alias", position)        the spec of the node at position, unchanged
          ("barrier", needs)         ready at once, when every (position, n) of needs has
                                     its first n elements (cat)
          ("pipelined", recipe(pf))  evaluate_recipe of recipe(pf)
        """
        value = self.node_order[idx]
        if type(value) is BlockArgument:
            return "trivial", None
        op = value.owner
        if type(op) in (ConstRefOp, cd.CimSaveStationaryOp):
            return "trivial", None
        plan = self._plan_at(idx, op)
        if plan is not None:
            return "pipelined", plan
        return self._passive_form(op)

    def _passive_form(self, op):
        """node_form of the ops with no cost of their own (no _SPEC_PLANNERS entry):
        shape ops pass their operand's spec through, embedding its index's, and cat is
        ready once all its inputs are."""
        if type(op) in SHAPE_OPS:
            return "alias", self.node_index[op.operand]
        if type(op) is cd.CimEmbeddingOp:
            # embedding: ready when its index (rhs) is; no cost
            return "alias", self.node_index[op.rhs]
        if type(op) is cd.CimCatOp:
            return "barrier", tuple((self.node_index[p], self._numel(p)) for p in op.tensors)
        raise NotImplementedError(f"TimingOnlyMapping: unsupported op {op}")

    def _build_spec_uncached(self, op, specs: dict):
        """Spec of the ops with no cost of their own (_passive_form)."""
        kind, arg = self._passive_form(op)
        if kind == "alias":
            return specs[arg]
        t = _latest_ready(specs, arg)
        return OutputGenSpec(start=t, first=t, interval=0.0, unit_size=float("inf"), count=1, end=t,
                             circuit_bound=False)

    def _plan_matvec(self, op: cd.CimMatvecOp):
        """Per row round: read the input vector, then one single-bit MAC per input bit
        (the lhs's bit width): from the RMU for a stationary rhs, from the WMU for a
        dynamic one.
        """
        lhs, rhs = op.lhs, op.rhs
        lhs_shape = self._shape(lhs)
        out_shape = self._shape(op.result)
        if len(self._shape(rhs)) == 3:
            # grouped: one round per copy = all nb blocks at once
            row_dims = lhs_shape[:-2]
            row_size = lhs_shape[-2] * lhs_shape[-1]
        else:
            row_dims = lhs_shape[:-1]
            row_size = lhs_shape[-1]
        total_rows = math.prod(row_dims)
        # everything one round produces: for a tiled matvec ([..., tiles, out]) every row
        # tile's partial vector, since the tile arrays run in parallel
        output_width = self._numel(op.result) // total_rows

        input_bits = bit_width(lhs)
        t = self.target_timing
        if type(_defining_op(rhs)) is cd.CimSaveStationaryOp:
            # stationary rhs: MAC from the RMU
            hardware_interval = t.wmu_read() + t.rmu_mac(input_bits)
            setup = 0.0
        else:
            # dynamic rhs: MAC from the WMU
            if self.target.architecture != "coupled_fine_grained_rmu_wmu":
                raise NotImplementedError(f"matvec with a dynamic rhs: not modeled for {self.target.architecture}")
            # one WMU MAC per input bit; the WMU MAC is destructive, so once the input
            # row's MACs are done the tile's rows are rewritten (read the keep copy,
            # write the compute copy, write the keep copy back) for the next row. Row
            # tiles sit in separate arrays and run in parallel.
            tile_rows = min(lhs_shape[-1], self.target.geometry.wmu.rows)
            rewrite = tile_rows * (t.wmu_read() + 2 * t.wmu_write())
            hardware_interval = t.wmu_read() + input_bits * t.wmu_mac(1) + rewrite
            # setup: the rhs's producer writes one staging copy as it comes out
            # (xdsl_hardware.OperationMapping.rhs_hold_arrays); once the whole rhs is
            # there, its rows are read and written into every copy (compute and keep,
            # per parallel copy), all in separate arrays, in parallel
            setup = tile_rows * (t.wmu_read() + t.wmu_write())

        lhs_i, lhs_numel = self.node_index[lhs], self._numel(lhs)
        rhs_i, rhs_numel = self.node_index[rhs], self._numel(rhs)

        def recipe(pf):
            return matvec_recipe(
                lhs_i=lhs_i, lhs_numel=lhs_numel,
                rhs_i=rhs_i, rhs_numel=rhs_numel,
                row_size=row_size, total_rows=total_rows,
                output_width=output_width,
                hardware_interval=hardware_interval, pf=pf, setup=setup,
            )
        return recipe

    def _plan_reduce_vec(self, op: cd.CimReduceVecOp):
        parent = op.operand
        parent_op = _defining_op(parent)
        parent_shape = self._shape(parent)
        reduce_dim_raw = op.reduce_dim.value.data
        reduce_dim = reduce_dim_raw if reduce_dim_raw >= 0 else reduce_dim_raw + len(parent_shape)

        if (
            type(parent_op) is cd.CimMatvecOp
            and reduce_dim == len(parent_shape) - 2
        ):
            return self._plan_reduce_vec_matvec_tiles(op, parent, parent_shape, reduce_dim)

        return self._plan_reduce_vec_logical(op, parent, parent_shape, reduce_dim)

    def _reduce_unit_cost(self, op: cd.CimReduceVecOp, reduce_n: int, reduce_w: int) -> float:
        """Cost of reducing one reduce_n-wide chunk to one element. "tree": one pass of
        the reduce tree. "vec_add": ceil(log2(reduce_n)) halving rounds on a
        reduce_n/2-wide vec_add circuit. Every pass reads its input vector from the WMU
        and writes its result back.
        """
        impl = _accel_impl_of(op, "reduce_impl")
        if impl == "tree":
            return self.target_timing.wmu_read() + self.target_timing.reduce_accel(reduce_n, reduce_w) + self.target_timing.wmu_write()
        if impl == "vec_add":
            half = reduce_n // 2
            rounds = math.ceil(math.log2(reduce_n))
            return rounds * (self.target_timing.wmu_read() + self.target_timing.vecadd_accel(half, reduce_w) + self.target_timing.wmu_write())
        raise ValueError(f"reduce_vec: unknown reduce_impl {impl!r} (expected 'tree' or 'vec_add')")

    def _plan_reduce_vec_matvec_tiles(self, op: cd.CimReduceVecOp, parent, parent_shape, reduce_dim):
        reduce_size = parent_shape[reduce_dim]
        output_width = self._shape(op.result)[-1]
        num_rows = self._numel(op.result) // output_width
        parent_i = self.node_index[parent]

        reduce_n = _accel_width_of(op, "reduce_width")
        reduce_w = _accel_width_of(op, "accel_data_width")
        per_pass_cost = self._reduce_unit_cost(op, reduce_n, reduce_w) * math.ceil(reduce_size / reduce_n)

        def recipe(pf):
            return reduce_vec_matvec_tiles_recipe(
                parent_i=parent_i, reduce_size=reduce_size, output_width=output_width,
                num_rows=num_rows, per_pass_cost=per_pass_cost, pf=pf,
            )
        return recipe

    def _plan_reduce_vec_logical(self, op: cd.CimReduceVecOp, parent, parent_shape, reduce_dim):
        reduce_size = parent_shape[reduce_dim]
        inner = math.prod(parent_shape[reduce_dim + 1:])
        num_problems = self._numel(parent) // reduce_size
        reduce_n = _accel_width_of(op, "reduce_width")
        reduce_w = _accel_width_of(op, "accel_data_width")
        hardware_interval = self._reduce_unit_cost(op, reduce_n, reduce_w) * math.ceil(reduce_size / reduce_n)

        parent_i = self.node_index[parent]

        def recipe(pf):
            return reduce_axis_recipe(
                parent_i=parent_i, reduce_size=reduce_size, inner=inner,
                num_problems=num_problems, outputs_per_problem=1,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_topk(self, op):
        """argmax / max: top-k over the last axis, k sequential comparator-tree passes
        per problem; each problem outputs k values.
        """
        parent = op.operand
        parent_shape = self._shape(parent)
        reduce_dim_raw = op.reduce_dim.value.data if op.reduce_dim is not None else -1
        reduce_dim = reduce_dim_raw if reduce_dim_raw >= 0 else reduce_dim_raw + len(parent_shape)
        if reduce_dim != len(parent_shape) - 1:
            raise NotImplementedError(
                f"TimingOnlyMapping: {type(op).__name__} unsupported reduce_dim={reduce_dim_raw} "
                "(only last-axis reduction is supported)"
            )

        reduce_size = parent_shape[reduce_dim]
        k = op.k.value.data if op.k is not None else 1
        num_problems = self._numel(op.result) // k
        argmax_n = _accel_width_of(op, "argmax_width")
        argmax_w = _accel_width_of(op, "accel_data_width")
        hardware_interval = (
            k * (self.target_timing.wmu_read() + self.target_timing.argmax_accel(argmax_n, argmax_w) + self.target_timing.wmu_write())
            * math.ceil(reduce_size / argmax_n)
        )
        parent_i = self.node_index[parent]

        def recipe(pf):
            return reduce_axis_recipe(
                parent_i=parent_i, reduce_size=reduce_size, inner=1,
                num_problems=num_problems, outputs_per_problem=k,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_scatter(self, op: cd.CimScatterOp):
        """Scatter: one vector write per index element."""
        index_i, index_numel = self.node_index[op.index], self._numel(op.index)
        src_i, src_numel = self.node_index[op.src], self._numel(op.src)
        hardware_interval = self.target_timing.wmu_write()

        def recipe(pf):
            return vector_op_recipe(
                a_i=index_i, a_numel=index_numel, b_i=src_i, b_numel=src_numel,
                vector_size=1, total_elements=index_numel, hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_lut_search(self, op):
        """lut_search (one value per index) and lut_search_vector (a whole row per
        index): pf counts lookups.
        """
        index = op.index if type(op) is cd.CimLutSearchOp else op.rhs
        index_i = self.node_index[index]
        total_lookups = self._numel(index)
        output_width = self._numel(op.result) // total_lookups
        hardware_interval = self.target_timing.rmu_read() + self.target_timing.wmu_read()

        def recipe(pf):
            return lut_search_recipe(
                index_i=index_i, total_lookups=total_lookups, output_width=output_width,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_vec_el_mul(self, op: cd.CimVecElMulOp):
        a_value, b_value = op.lhs, op.rhs
        if b_value is None:
            raise NotImplementedError(
                "TimingOnlyMapping: vec_el_mul with a literal rhs_scalar reached the scheduling backend -- "
                "convertStaticValuesToStationary should have eliminated this before cim IR ever reaches here"
            )
        return self._vector_op_recipe(op, a_value, b_value)

    def _vector_op_recipe(self, op, a_value, b_value):
        """Shared recipe(pf) of vec_el_mul, vec_add and vec_select: pf counts 1-lane units
        (shared_lane_plan, from lane_costs: a round is a WMU read, one unit delay and a
        WMU write)."""
        a_i, a_numel = self.node_index[a_value], self._numel(a_value)
        b_i, b_numel = self.node_index[b_value], self._numel(b_value)
        vector_size = self._shape(op.results[0])[-1]
        total_elements = self._numel(op.results[0])
        costs = lane_costs(op, self.target_timing)

        def recipe(pf):
            in_flight, interval, _ = shared_lane_plan(pf, *costs)
            return vector_op_recipe(
                a_i=a_i, a_numel=a_numel,
                b_i=b_i, b_numel=b_numel,
                vector_size=vector_size, total_elements=total_elements,
                hardware_interval=interval, pf=in_flight,
            )
        return recipe

    def _plan_vec_add(self, op: cd.CimVecAddOp):
        a_value, b_value = op.lhs, op.rhs
        if b_value is None:
            raise NotImplementedError(
                "TimingOnlyMapping: vec_add with a literal rhs_scalar reached the scheduling backend -- "
                "convertStaticValuesToStationary should have eliminated this before cim IR ever reaches here"
            )
        return self._vector_op_recipe(op, a_value, b_value)

    def _plan_vec_select(self, op: cd.CimVecSelectOp):
        """vec_select (torch.where with one fill word), as vec_add on 1-lane multiplexers
        (_vector_op_recipe). The condition is stored."""
        return self._vector_op_recipe(op, op.operand, op.cond)

    def _plan_row_absmax(self, op: cd.CimRowAbsmaxOp):
        """row_absmax: max |x| over each row of row_width values, as a tree reduce_vec:
        ceil(row_width / N) passes of the N-input abs-max tree (token_quant), each a WMU
        read, a tree pass and a WMU write; pf = rows in parallel."""
        row_width = self._shape(op.operand)[-1]
        num_rows = self._numel(op.result)
        w = _accel_width_of(op, "accel_data_width")
        n = ABSMAX_TREE_INPUTS
        t = self.target_timing
        hardware_interval = math.ceil(row_width / n) * (t.wmu_read() + t.absmax_tree(n, w) + t.wmu_write())
        operand_i = self.node_index[op.operand]

        def recipe(pf):
            return reduce_axis_recipe(
                parent_i=operand_i, reduce_size=row_width, inner=1,
                num_problems=num_rows, outputs_per_problem=1,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_quantize_per_token(self, op: cd.CimQuantizePerTokenOp):
        """quantize_per_token: one row per round, as softmax's row pipeline: a WMU read,
        the row's scale (token_scale, once per row), the per-element rounding
        (quant_mant_round, one element delay: its lanes are shared over the round,
        xdsl_accel_concurrency) and a WMU write; pf counts the rounding lanes
        (shared_lane_plan). A row starts once both its values and its row max are ready."""
        costs = lane_costs(op, self.target_timing)
        num_rows, row_width = costs[0], costs[1]
        x_i, max_i = self.node_index[op.operand], self.node_index[op.row_max]

        def recipe(pf):
            in_flight, interval, _ = shared_lane_plan(pf, *costs)
            return PipelineRecipe(
                first_needs=((x_i, row_width), (max_i, 1)),
                all_needs=((x_i, num_rows * row_width), (max_i, num_rows)),
                num_problems=num_rows,
                batch_terms=((x_i, 1, row_width, 1), (max_i, 1, 1, 1)),
                outputs_per_problem=row_width, hardware_interval=interval, pf=in_flight, setup=0.0,
            )
        return recipe

    def _plan_pointwise(self, op):
        """Pointwise accelerator op (relu, gelu, silu, elu, leaky_relu, quantize,
        dequantize, rsqrt): one evaluation per element, read from and written back to
        the WMU; the ops differ only in the circuit.
        """
        t = self.target_timing
        w = _accel_width_of(op, "accel_data_width")
        if type(op) is cd.CimReluOp:
            circuit = t.relu_accel(w)
        elif type(op) is cd.CimRsqrtOp:
            circuit = t.rsqrt_cmos(w)
        elif type(op) is cd.CimGeluOp:
            circuit = t.gelu_cmos(op.variant.data, w)
        elif type(op) is cd.CimSiluOp:
            circuit = t.silu_cmos(w)
        elif type(op) is cd.CimEluOp:
            circuit = t.elu_cmos(w, float_from_prop(op.alpha))
        elif type(op) is cd.CimLeakyReluOp:
            circuit = t.leaky_relu_accel(w, float_from_prop(op.negative_slope))
        elif type(op) is cd.CimQuantizeOp:
            mw = int_from_prop(op.mantissa_width) if op.mantissa_width is not None else None
            circuit = t.quantize_cmos(op.variant.data, w, mw)
        else:  # CimDequantizeOp
            mw = int_from_prop(op.mantissa_width) if op.mantissa_width is not None else None
            circuit = t.dequantize_cmos(op.variant.data, w, mw)
        hardware_interval = t.wmu_read() + circuit + t.wmu_write()
        operand_i = self.node_index[op.operand]
        total_elements = self._numel(op.result)

        def recipe(pf):
            return lut_search_recipe(
                index_i=operand_i, total_lookups=total_elements, output_width=1,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    def _plan_row_op(self, op):
        """Row accelerator op (softmax, layer_norm, rms_norm): one invocation handles one
        row. softmax is the hierarchical row pipeline for the real row width
        (load_results.softmax_row_pipeline_ns), its pf counting the lanes of its
        per-element stages (shared_lane_plan); layer_norm / rms_norm rows are hidden_size
        wide, pf = rows in parallel. A layer_norm circuit streams its row ln_lanes
        elements per cycle (load_results.layer_norm_accel_ns: cycles_per_row x clock);
        rms_norm's is characterized at N=4 only (other sizes raise in load_results).
        """
        t = self.target_timing
        operand_i = self.node_index[op.operand]
        if type(op) is cd.CimSoftmaxOp:
            costs = lane_costs(op, t)

            def softmax_recipe(pf):
                in_flight, interval, _ = shared_lane_plan(pf, *costs)
                return row_op_recipe(
                    parent_i=operand_i, row_width=costs[1], num_rows=costs[0],
                    hardware_interval=interval, pf=in_flight,
                )
            return softmax_recipe
        else:
            row_width = int_from_prop(op.hidden_size)
            w = _accel_width_of(op, "accel_data_width")
            if type(op) is cd.CimLayerNormOp:
                # the streamed circuit: ln_lanes lanes per cycle over the whole row
                circuit = t.layer_norm_cmos(op.variant.data, row_width, w, _accel_width_of(op, "ln_lanes"))
            else:  # CimRMSNormOp
                circuit = t.rms_norm_cmos(row_width, w)
        hardware_interval = t.wmu_read() + circuit + t.wmu_write()
        num_rows = self._numel(op.result) // row_width

        def recipe(pf):
            return row_op_recipe(
                parent_i=operand_i, row_width=row_width, num_rows=num_rows,
                hardware_interval=hardware_interval, pf=pf,
            )
        return recipe

    # op type -> _plan_* method; _build_spec caches each node's plan
    _SPEC_PLANNERS = {
        cd.CimMatvecOp: _plan_matvec,
        cd.CimReduceVecOp: _plan_reduce_vec,
        cd.CimLutSearchOp: _plan_lut_search,
        cd.CimLutSearchVectorOp: _plan_lut_search,
        cd.CimVecElMulOp: _plan_vec_el_mul,
        cd.CimVecAddOp: _plan_vec_add,
        cd.CimArgmaxOp: _plan_topk,
        cd.CimMaxOp: _plan_topk,
        cd.CimScatterOp: _plan_scatter,
        cd.CimReluOp: _plan_pointwise,
        cd.CimRsqrtOp: _plan_pointwise,
        cd.CimGeluOp: _plan_pointwise,
        cd.CimSiluOp: _plan_pointwise,
        cd.CimEluOp: _plan_pointwise,
        cd.CimLeakyReluOp: _plan_pointwise,
        cd.CimQuantizeOp: _plan_pointwise,
        cd.CimDequantizeOp: _plan_pointwise,
        cd.CimSoftmaxOp: _plan_row_op,
        cd.CimLayerNormOp: _plan_row_op,
        cd.CimRMSNormOp: _plan_row_op,
        cd.CimVecSelectOp: _plan_vec_select,
        cd.CimRowAbsmaxOp: _plan_row_absmax,
        cd.CimQuantizePerTokenOp: _plan_quantize_per_token,
    }


def _defining_op(value: SSAValue):
    """The op defining `value`; None for a block argument (type() check: faster than isinstance())."""
    return None if type(value) is BlockArgument else value.owner


def _is_return(op) -> bool:
    return isinstance(op, ReturnOp)
