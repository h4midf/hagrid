"""How many accelerator circuits of each kind a mapped schedule needs, and their area
and power. Read-only: it measures a computed schedule, never changes it.

Each accelerator op holds its circuits (accelerator_demands: {sub-kind: count}) over
its [start, end) window. A sub-kind's count is the peak of the circuits in use at
once (sweep-line, ties end-before-start: a circuit freed at t can be taken at t), so
ops that do not overlap in time share circuits. Reduce trees also borrow across
sizes: a reduction can run on any idle larger tree (_reduce_tree_counts). Timing
still uses each op's own tree. vec_add, vec_el_mul and vec_select are built from 1-lane
units; their pf counts them (xdsl_timing._vector_op_recipe). A softmax op holds each stage of its row
pipeline separately (_softmax_demands): its adder, multiplier and sum tree count in
the vec_add, vec_el_mul and reduce tree pools; its other stages are softmax's own.
compiler/xdsl_search.py uses the same numbers for its optional accelerator area
budgets.
"""
import functools
import math

from xdsl.ir import SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.rewrite_helpers import shape_of
from custom_op_library.common.tools import load_results
from targets.target_support import TARGET, timing_of

# the target's own array reads / writes, the same the timing model uses
_TIMING = timing_of(TARGET)

# cim op -> its accelerator circuit kind. The other ops use the arrays (matvec,
# lut_search) or no circuit at all (layout ops).
ACCEL_OP_KINDS = {
    cd.CimReduceVecOp: "reduce_tree",
    cd.CimArgmaxOp: "argmax_tree",
    # max is timed on the argmax tree (xdsl_timing._plan_topk), so it runs on one
    cd.CimMaxOp: "argmax_tree",
    cd.CimVecAddOp: "vec_add",
    cd.CimVecElMulOp: "vec_el_mul",
    cd.CimReluOp: "relu",
    cd.CimRsqrtOp: "rsqrt",
    cd.CimGeluOp: "gelu",
    cd.CimSoftmaxOp: "softmax",
    # the streamed layer-norm circuit (one per row in flight)
    cd.CimLayerNormOp: "layer_norm",
    cd.CimQuantizeOp: "quantize",
    cd.CimDequantizeOp: "dequantize",
    cd.CimSiluOp: "silu",
    cd.CimEluOp: "elu",
    cd.CimLeakyReluOp: "leaky_relu",
    # per-token int8 quantisation and the select (kernels/gpt2.py)
    cd.CimRowAbsmaxOp: "absmax_tree",
    cd.CimQuantizePerTokenOp: "quantize_per_token",
    cd.CimVecSelectOp: "vec_select",
}


_REDUCE_TREE_PREFIX = "reduce_tree_n"
_SOFTMAX_PREFIX = "softmax_"
# quantize_per_token's two circuits: token_scale (one per row in flight) and
# quant_mant_round (lanes shared over the round)
_TOKEN_QUANT_PREFIX = "quantize_per_token_"
_TOKEN_SCALE_SUB_KIND = f"{_TOKEN_QUANT_PREFIX}token_scale"
_TOKEN_ROUND_SUB_KIND = f"{_TOKEN_QUANT_PREFIX}quant_mant_round"


def _accel_sub_kind(op) -> str:
    """The circuit a non-softmax accelerator op runs on, counted and priced
    separately: its ACCEL_OP_KINDS kind, with reduce_tree split by tree inputs."""
    if isinstance(op, cd.CimReduceVecOp):
        return f"{_REDUCE_TREE_PREFIX}{op.attributes['reduce_width'].value.data}"
    return ACCEL_OP_KINDS[type(op)]


def _kind_of_sub_kind(sub_kind: str) -> str:
    if sub_kind.startswith(_REDUCE_TREE_PREFIX):
        return "reduce_tree"
    if sub_kind.startswith(_SOFTMAX_PREFIX):
        return "softmax"
    if sub_kind.startswith(_TOKEN_QUANT_PREFIX):
        return "quantize_per_token"
    return sub_kind


# kinds reported per sub-kind, not as one key: reduce_tree by tree size, softmax by
# stage (only the stages that are not pooled with another kind), quantize_per_token by
# circuit
_SPLIT_KINDS = ("softmax", "reduce_tree", "quantize_per_token")


def accelerator_demands(op, pf: int) -> dict[str, int]:
    """{sub-kind: circuits} an accelerator op holds at once. Its pf (circuits in
    parallel), at most its number of independent problems (_instance_cap); for the ops
    on shared 1-lane units (xdsl_timing.LANE_OPS: vec_add, vec_el_mul, vec_select,
    softmax, quantize_per_token), whose pf counts lanes, the lanes and once-per-row
    circuits of xdsl_timing.shared_lane_plan, the same plan the timing uses."""
    if isinstance(op, cd.CimSoftmaxOp):
        return _softmax_demands(op, pf)
    if isinstance(op, cd.CimQuantizePerTokenOp):
        return _token_quant_demands(op, pf)
    if isinstance(op, (cd.CimVecAddOp, cd.CimVecElMulOp, cd.CimVecSelectOp)):
        (lanes,) = _lane_plan(op, pf)[2]
        return {_accel_sub_kind(op): lanes}
    return {_accel_sub_kind(op): min(pf, _instance_cap(op))}


def _lane_plan(op, pf: int) -> tuple:
    """xdsl_timing.shared_lane_plan of a LANE_OPS op at pf, on this target's timing."""
    from compiler.xdsl_timing import lane_costs, shared_lane_plan

    return shared_lane_plan(pf, *lane_costs(op, _TIMING))


def _token_quant_plan(op) -> tuple:
    """(round_ns, token_scale row, quant_mant_round row with its lane register) of a
    quantize_per_token row: the round is WMU read + token_scale + quant_mant_round + WMU
    write, as in xdsl_timing._plan_quantize_per_token."""
    from compiler.xdsl_timing import TOKEN_QUANT_MANTISSA_WIDTH as mw

    w = op.attributes["accel_data_width"].value.data
    round_ns = (_TIMING.wmu_read() + _TIMING.token_scale(w, mw) + _TIMING.quantize_cmos("mant_round", w, mw)
                + _TIMING.wmu_write())
    scale_row = load_results.token_scale_area_power(w, mw)
    # a rounding lane's register: x, m, sh in, the int8 q out
    round_row = _with_lane_register(load_results.quantize_accel_area_power("mant_round", w, mw), w + mw + 6 + 8)
    return round_ns, scale_row, round_row


def _token_quant_demands(op, pf: int) -> dict[str, int]:
    """{sub-kind: circuits} of a quantize_per_token op at pf (_lane_plan): one
    token_scale per row in flight, and the rounding lanes."""
    in_flight, _, (lanes,) = _lane_plan(op, pf)
    return {_TOKEN_SCALE_SUB_KIND: in_flight, _TOKEN_ROUND_SUB_KIND: lanes}


def _softmax_stage_sub_kind(module: str, row: dict, data_width: int) -> str:
    """The pool a softmax stage counts in: the 1-lane adder and multiplier join
    vec_add and vec_el_mul when they are the same circuit, at the op's data width (the
    softmax pipelines are characterized at 32 bits only: at Q8.8 their 32-bit adder
    and multiplier are softmax's own), the sum tree joins the reduce trees of its size,
    every other stage is softmax's own."""
    if module in ("vec_add_vv", "vec_el_mul_vv_round") and row["params"]["W"] == data_width:
        return "vec_add" if module == "vec_add_vv" else "vec_el_mul"
    if module == "reduce_tree":
        return f"{_REDUCE_TREE_PREFIX}{row['params']['N']}"
    return f"{_SOFTMAX_PREFIX}{module}"


# register bits (input operands + result) of each per-element softmax stage's lane,
# from the stage's ports (custom_op_library/softmax/rtl, quark_shared/rtl)
_SOFTMAX_LANE_REGISTER_BITS = {
    "vec_add_vv": lambda p: 3 * p["W"],
    "vec_el_mul_vv_round": lambda p: 3 * p["W"],
    "exp_pwl16_standalone": lambda p: 24 + 16,
    "ibert_exp_pwl_standalone": lambda p: 24 + 16,
    "log2exp_standalone": lambda p: 24 + 4 + 9,
    "aldivision_standalone": lambda p: 4 + p["SUM_W"] + 16,
    "gqa_lut_exp8": lambda p: 2 * p["W"],
    "quark_appro_exp": lambda p: 2 * p["W"],
}


def _with_lane_register(row: dict, bits: int) -> dict:
    """A shared 1-lane unit's row with its register (load_results.lane_register_area_power)
    added to its area and power; the delay stays the lane's own."""
    reg = load_results.lane_register_area_power(bits)
    return {**row, **{field: row[field] + reg[field]
                      for field in ("area_um2_45nm", "area_um2_28nm", "leakage_uW", "dynamic_mW_est")}}


@functools.lru_cache(maxsize=None)
def softmax_data_width(op) -> int:
    """A softmax op's data width: its operand's fixed-point width."""
    elem = op.operand.type.element_type
    if not isinstance(elem, cd.FixedType):
        raise TypeError(f"softmax operand is {elem}, not fixed point")
    return elem.width.data


def _softmax_plan(variant: str, row_width: int, data_width: int) -> tuple:
    """(round_ns, [(sub-kind, row, per_element)]) of a softmax row: the round is WMU
    read + row pipeline + WMU write, as in xdsl_timing._plan_row_op. A per-element
    stage's row includes its lane register."""
    round_ns = _TIMING.wmu_read() + load_results.softmax_row_pipeline_ns(variant, row_width) + _TIMING.wmu_write()
    stages = []
    for _, module, row, per_element, _ in load_results.softmax_stages(variant, row_width):
        sub_kind = _softmax_stage_sub_kind(module, row, data_width)
        if per_element:
            if module not in _SOFTMAX_LANE_REGISTER_BITS:
                raise NotImplementedError(f"softmax {variant!r}: no lane register width for stage {module!r}")
            row = _with_lane_register(row, _SOFTMAX_LANE_REGISTER_BITS[module](row["params"]))
        stages.append((sub_kind, row, per_element))
    return round_ns, tuple(stages)


def _softmax_demands(op, pf: int) -> dict[str, int]:
    """{sub-kind: circuits} of a softmax op at pf (_lane_plan): a once-per-row stage (the
    trees, the reciprocal) holds one circuit per row in flight, a per-element stage its
    lanes."""
    in_flight, _, lanes = _lane_plan(op, pf)
    _, stages = _softmax_plan(op.variant.data, shape_of(op.result.type)[-1], softmax_data_width(op))
    per_element_lanes = iter(lanes)
    demands: dict[str, int] = {}
    for sub_kind, _row, per_element in stages:
        count = next(per_element_lanes) if per_element else in_flight
        demands[sub_kind] = demands.get(sub_kind, 0) + count
    return demands


def _instance_cap(op) -> int:
    """An op's independent problems, the most circuits it can use at once: one per
    output element (reduce, pointwise ops), per top-k (argmax), per vector (vec_add,
    vec_el_mul) or per row (softmax). From the IR shape only. Raises for an op
    outside ACCEL_OP_KINDS."""
    if isinstance(op, cd.CimReduceVecOp):
        cap = max(1, _numel(op))
    elif isinstance(op, (cd.CimArgmaxOp, cd.CimMaxOp)):
        k = int(op.k.value.data) if op.k is not None else 1
        cap = max(1, _numel(op) // max(1, k))
    elif isinstance(op, (cd.CimVecAddOp, cd.CimVecElMulOp, cd.CimVecSelectOp)):
        shape = shape_of(op.result.type)
        vector_size = shape[-1] if shape else 1
        cap = max(1, _numel(op) // max(1, int(vector_size)))
    elif isinstance(op, (cd.CimReluOp, cd.CimGeluOp, cd.CimQuantizeOp, cd.CimDequantizeOp, cd.CimSiluOp,
                         cd.CimEluOp, cd.CimLeakyReluOp, cd.CimRsqrtOp)):
        cap = max(1, _numel(op))
    elif isinstance(op, cd.CimRowAbsmaxOp):
        # one value per row
        cap = max(1, _numel(op))
    elif isinstance(op, (cd.CimSoftmaxOp, cd.CimQuantizePerTokenOp, cd.CimLayerNormOp)):
        shape = shape_of(op.result.type)
        row_width = shape[-1] if shape else 1
        cap = max(1, _numel(op) // max(1, int(row_width)))
    else:
        raise ValueError(f"_instance_cap: {type(op).__name__} is not one of ACCEL_OP_KINDS")
    return cap


def _numel(op) -> int:
    shape = shape_of(op.result.type)
    return math.prod(shape) if shape else 1


def _peak_instances(intervals: list[tuple[float, float, int]]) -> int:
    events = []
    for start, end, count in intervals:
        if end <= start:
            continue
        events.extend(((end, 0, -count), (start, 1, count)))
    current = peak = 0
    for _, _, delta in sorted(events):
        current += delta
        peak = max(peak, current)
    return peak


def _reduce_tree_counts(intervals_by_size: dict[int, list[tuple[float, float, int]]]) -> dict[int, int]:
    """{tree inputs: trees provisioned}. Walks the sizes from largest to smallest; at
    every moment a size's demand first takes the idle larger trees, and the size gets
    only as many trees of its own as the remaining demand peaks at. Its own idle trees
    then join the pool the smaller sizes borrow from."""
    times = sorted({t for ivs in intervals_by_size.values() for start, end, _ in ivs if end > start
                    for t in (start, end)})
    position = {t: i for i, t in enumerate(times)}
    segments = max(0, len(times) - 1)
    spare = [0] * segments
    counts = {}
    for size in sorted(intervals_by_size, reverse=True):
        delta = [0] * (segments + 1)
        for start, end, count in intervals_by_size[size]:
            if end > start:
                delta[position[start]] += count
                delta[position[end]] -= count
        demand = []
        running = 0
        for i in range(segments):
            running += delta[i]
            demand.append(running)
        own_needed = [max(0, d - s) for d, s in zip(demand, spare)]
        provisioned = max(own_needed, default=0)
        counts[size] = provisioned
        spare = [s - min(d, s) + provisioned - n for d, s, n in zip(demand, spare, own_needed)]
    return counts


def _provisioned_counts(intervals: dict[str, list[tuple[float, float, int]]]) -> dict[str, int]:
    """{sub-kind: circuits provisioned}: each sub-kind's peak (_peak_instances), except
    the reduce trees, which borrow across sizes (_reduce_tree_counts)."""
    counts = {}
    trees = {}
    for sub_kind, ivs in intervals.items():
        if sub_kind.startswith(_REDUCE_TREE_PREFIX):
            trees[int(sub_kind[len(_REDUCE_TREE_PREFIX):])] = ivs
        else:
            counts[sub_kind] = _peak_instances(ivs)
    for size, count in _reduce_tree_counts(trees).items():
        counts[f"{_REDUCE_TREE_PREFIX}{size}"] = count
    return counts


def compute_accel_peak_concurrency(block, specs: dict[int, object], mapped_pf: dict[int, int],
                                   node_index: dict) -> dict[str, int]:
    """`specs` and `mapped_pf` are a TimingOnlyMapping's `specs` and `mapped_pf`,
    keyed by node position (`node_index`). Returns {kind: circuits provisioned}
    (_provisioned_counts) for every kind in ACCEL_OP_KINDS except the split ones
    (_SPLIT_KINDS); a kind no op uses gets 0. A split kind reports one key per sub-kind
    present instead ("reduce_tree_n8", "softmax_exp_pwl16_standalone", ...)."""
    intervals: dict[str, list] = {kind: [] for kind in ACCEL_OP_KINDS.values() if kind not in _SPLIT_KINDS}
    for op in block.ops:
        if type(op) not in ACCEL_OP_KINDS or not op.results:
            continue
        idx = node_index[op.results[0]]
        spec = specs.get(idx)
        if spec is None:
            continue
        for sub_kind, count in accelerator_demands(op, mapped_pf[idx]).items():
            intervals.setdefault(sub_kind, []).append((float(spec.start), float(spec.end), count))

    return _provisioned_counts(intervals)


def accel_unit_rows(op) -> dict[str, dict]:
    """{sub-kind: row} of every circuit an accelerator op uses (accelerator_demands'
    keys): one per softmax stage, else accel_unit_row."""
    if isinstance(op, cd.CimSoftmaxOp):
        _, stages = _softmax_plan(op.variant.data, shape_of(op.result.type)[-1], softmax_data_width(op))
        return {sub_kind: row for sub_kind, row, _ in stages}
    if isinstance(op, cd.CimQuantizePerTokenOp):
        _, scale_row, round_row = _token_quant_plan(op)
        return {_TOKEN_SCALE_SUB_KIND: scale_row, _TOKEN_ROUND_SUB_KIND: round_row}
    return {_accel_sub_kind(op): accel_unit_row(op)}


def accel_unit_row(op) -> dict:
    """area_um2_45nm / leakage_uW / dynamic_mW_est of one circuit of this non-softmax
    op, chosen from the op's stamped attributes (variant, width, impl): the same
    circuit the timing model (compiler/xdsl_timing.py) uses. `op` must be an
    ACCEL_OP_KINDS op."""
    from compiler.xdsl_timing import _accel_impl_of, _accel_width_of
    from custom_op_library.common.tools import load_results

    if isinstance(op, cd.CimReduceVecOp):
        n = _accel_width_of(op, "reduce_width")
        w = _accel_width_of(op, "accel_data_width")
        impl = _accel_impl_of(op, "reduce_impl")
        if impl == "tree":
            return load_results.reduce_accel_area_power(n, w)
        if impl == "vec_add":
            # one vec_add circuit of n/2 lanes (the widest round), reused over the
            # log2(n) halving rounds
            return load_results.vecadd_accel_area_power(max(1, n // 2), w)
        raise ValueError(f"accel_unit_row: unknown reduce_impl {impl!r} (expected 'tree' or 'vec_add')")
    if isinstance(op, (cd.CimArgmaxOp, cd.CimMaxOp)):
        n = _accel_width_of(op, "argmax_width")
        w = _accel_width_of(op, "accel_data_width")
        return load_results.argmax_accel_area_power(n, w)
    if isinstance(op, cd.CimVecAddOp):
        w = _accel_width_of(op, "accel_data_width")
        # one 1-lane adder and its register (two operands, one result)
        return _with_lane_register(load_results.vecadd_accel_area_power(1, w), 3 * w)
    if isinstance(op, cd.CimVecElMulOp):
        w = _accel_width_of(op, "accel_data_width")
        return _with_lane_register(load_results.vecmul_accel_area_power(1, w), 3 * w)
    if isinstance(op, cd.CimVecSelectOp):
        w = _accel_width_of(op, "accel_data_width")
        # one 1-lane multiplexer and its register (x, the fill word, the result, the mask bit)
        return _with_lane_register(load_results.vecselect_accel_area_power(1, w), 3 * w + 1)
    if isinstance(op, cd.CimRowAbsmaxOp):
        from compiler.xdsl_timing import ABSMAX_TREE_INPUTS
        w = _accel_width_of(op, "accel_data_width")
        return load_results.absmax_tree_area_power(ABSMAX_TREE_INPUTS, w)
    if isinstance(op, cd.CimReluOp):
        w = _accel_width_of(op, "accel_data_width")
        return load_results.relu_accel_area_power(w)
    if isinstance(op, cd.CimRsqrtOp):
        w = _accel_width_of(op, "accel_data_width")
        return load_results.rsqrt_accel_area_power(w)
    if isinstance(op, cd.CimSiluOp):
        w = _accel_width_of(op, "accel_data_width")
        return load_results.silu_accel_area_power(w)
    if isinstance(op, cd.CimGeluOp):
        w = _accel_width_of(op, "accel_data_width")
        return load_results.gelu_accel_area_power(op.variant.data, w)
    if isinstance(op, cd.CimLayerNormOp):
        from compiler.dialects.attr_helpers import int_from_prop
        w = _accel_width_of(op, "accel_data_width")
        return load_results.layer_norm_accel_area_power(op.variant.data, int_from_prop(op.hidden_size), w,
                                                        _accel_width_of(op, "ln_lanes"))
    if isinstance(op, (cd.CimQuantizeOp, cd.CimDequantizeOp)):
        from compiler.dialects.attr_helpers import int_from_prop
        w = _accel_width_of(op, "accel_data_width")
        mw = int_from_prop(op.mantissa_width) if op.mantissa_width is not None else None
        fn = load_results.quantize_accel_area_power if isinstance(op, cd.CimQuantizeOp) \
            else load_results.dequantize_accel_area_power
        return fn(op.variant.data, w, mw)
    if isinstance(op, (cd.CimEluOp, cd.CimLeakyReluOp)):
        from compiler.dialects.attr_helpers import float_from_prop
        w = _accel_width_of(op, "accel_data_width")
        if isinstance(op, cd.CimEluOp):
            alpha = float_from_prop(op.alpha) if op.alpha is not None else 1.0
            return load_results.elu_accel_area_power(w, alpha)
        slope = float_from_prop(op.negative_slope) if op.negative_slope is not None else 0.01
        return load_results.leaky_relu_accel_area_power(w, slope)
    raise ValueError(f"accel_unit_row: {type(op).__name__} is not one of ACCEL_OP_KINDS")


def per_kind_accel_rows(block) -> dict[str, dict]:
    """{sub-kind: row} for every circuit the block uses (accel_unit_rows). All ops of
    one sub-kind, including the softmax stages pooled into it, must run on the same
    circuit; raises if two disagree on its area."""
    per_kind_row: dict[str, dict] = {}
    for op in block.ops:
        if type(op) not in ACCEL_OP_KINDS:
            continue
        for sub_kind, row in accel_unit_rows(op).items():
            existing = per_kind_row.get(sub_kind)
            if existing is None:
                per_kind_row[sub_kind] = row
            elif abs(existing["area_um2_45nm"] - row["area_um2_45nm"]) > 1e-6:
                raise ValueError(f"block has {sub_kind!r} circuits with disagreeing per-instance area: "
                                 f"{existing['area_um2_45nm']} vs {row['area_um2_45nm']}")
    return per_kind_row


def combine_accel_peak_area_power(peak_concurrency: dict[str, int], per_kind_row: dict[str, dict]) -> dict:
    """The chip's accelerator area and power: every sub-kind's per-circuit area,
    leakage and dynamic power times its peak count, per sub-kind and summed."""
    total_area_um2 = 0.0
    total_leakage_uw = 0.0
    total_dynamic_mw = 0.0
    per_kind = {}
    for kind, peak_count in peak_concurrency.items():
        row = per_kind_row.get(kind)
        area = row["area_um2_45nm"] * peak_count if row else 0.0
        leakage = row["leakage_uW"] * peak_count if row else 0.0
        dynamic = row["dynamic_mW_est"] * peak_count if row else 0.0
        total_area_um2 += area
        total_leakage_uw += leakage
        total_dynamic_mw += dynamic
        per_kind[kind] = {
            "peak_count": peak_count, "area_um2_45nm": area, "leakage_uW": leakage, "dynamic_mW_est": dynamic,
        }
    return {
        "total_area_um2": total_area_um2, "total_leakage_uW": total_leakage_uw,
        "total_dynamic_mW": total_dynamic_mw, "per_kind": per_kind,
    }


def total_accel_peak_area_power(block, specs: dict[int, object], mapped_pf: dict[int, int], node_index: dict) -> dict:
    """combine_accel_peak_area_power of the schedule's peak counts
    (compute_accel_peak_concurrency) and per-circuit rows (per_kind_accel_rows)."""
    peak = compute_accel_peak_concurrency(block, specs, mapped_pf, node_index)
    return combine_accel_peak_area_power(peak, per_kind_accel_rows(block))


def accel_area_index(block, node_index: dict) -> tuple:
    """Everything accel_kinds_peak_area_um2 needs from `block` that never
    changes during a search, computed once: (entries, per_kind_row),
    where entries is [(node position, op)] for every ACCEL_OP_KINDS op
    (block order) and per_kind_row is per_kind_accel_rows(block)."""
    entries = []
    for op in block.ops:
        if type(op) not in ACCEL_OP_KINDS or not op.results:
            continue
        entries.append((node_index[op.results[0]], op))
    return entries, per_kind_accel_rows(block)


def accel_kinds_peak_area_um2(index: tuple, specs: dict[int, object], mapped_pf: dict[int, int], kinds) -> dict[str, float]:
    """{kind: peak area (um^2)} for every kind in `kinds`, from one pass over the
    block's accelerator ops and a precomputed accel_area_index. A kind's area is the
    sum over its sub-kinds (including the softmax stages pooled into it) of each
    sub-kind's provisioned count (_provisioned_counts) times its per-instance area."""
    entries, per_kind_row = index
    intervals: dict[str, dict[str, list]] = {kind: {} for kind in kinds}
    for idx, op in entries:
        spec = specs.get(idx)
        if spec is None:
            continue
        for sub_kind, count in accelerator_demands(op, mapped_pf[idx]).items():
            by_sub_kind = intervals.get(_kind_of_sub_kind(sub_kind))
            if by_sub_kind is None:
                continue
            by_sub_kind.setdefault(sub_kind, []).append((float(spec.start), float(spec.end), count))
    areas = {}
    for kind in kinds:
        areas[kind] = sum(per_kind_row[sub_kind]["area_um2_45nm"] * count
                          for sub_kind, count in _provisioned_counts(intervals[kind]).items())
    return areas


def accel_kind_peak_area_um2(block, specs: dict[int, object], mapped_pf: dict[int, int], node_index: dict,
                             kind: str) -> float:
    """Peak area (um^2) of one ACCEL_OP_KINDS kind alone (e.g. "reduce_tree"), for a
    per-kind area budget in compiler/xdsl_search.py."""
    return accel_kinds_peak_area_um2(accel_area_index(block, node_index), specs, mapped_pf, (kind,))[kind]
