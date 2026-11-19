"""Stamps each accelerator op's circuit parameters: the lane width (argmax_width from
its config entry, looked up by fx_name; reduce_width from the IR, see
reduce_tree_inputs; vec_add and vec_el_mul have none, they are built from 1-lane
units), the circuit impl (reduce_impl),
and accel_data_width (the fixed-point width, 32 = Q12.20, 16 = Q8.8). The timing
model's accelerator functions take these (e.g. reduce_accel_ns(n, w)).

compiler/lowering.py runs it twice:
  after the cinm canon   stamps the cinm ops; copy_provenance carries the attributes
                         onto the ops they lower into (some cim ops, such as
                         cim.broadcast, get them only this way)
  after tiling           stamps the cim ops created after the cim lowering (the
                         cim.reduce_vec that sums a tiled matvec's row tiles)
"""
from xdsl.dialects.builtin import IntegerAttr, ModuleOp, StringAttr, i64

from compiler.dialects import cim_dialect as cd
from compiler.dialects import cinm_dialect as nd
from compiler.dialects.rewrite_helpers import shape_of
from compiler.ops.dispatch import fx_name_of
from custom_op_library.common.tools import load_results

# reduce ops: their tree's input count comes from the IR (reduce_tree_inputs), not
# from the config
_REDUCE_OP_TYPES = (nd.CinmReduceVecOp, cd.CimReduceVecOp)

# op type -> its lane-width config field, or None for an op whose circuit takes
# only the data width (vec_add, vec_el_mul, the activations and layer_norm; their
# other parameters, e.g. elu's alpha or layer_norm's hidden_size, are read off the op
# itself). cinm.scalar_add lowers to the same cim.vec_add.
_LANE_FIELD_OF_OP_TYPE = {
    nd.CinmReduceVecOp: None,
    nd.CinmVecElMulOp: None,
    nd.CinmVecAddOp: None,
    nd.CinmScalarAddOp: None,
    nd.CinmArgmaxOp: "argmax_width",
    nd.CinmMaxOp: "argmax_width",
    nd.CinmReluOp: None,
    nd.CinmRsqrtOp: None,
    nd.CinmGeluOp: None,
    nd.CinmSiluOp: None,
    nd.CinmEluOp: None,
    nd.CinmLeakyReluOp: None,
    # a layer_norm circuit streams its row ln_lanes elements per cycle
    nd.CinmLayerNormOp: "ln_lanes",
    cd.CimReduceVecOp: None,
    cd.CimVecElMulOp: None,
    cd.CimVecAddOp: None,
    cd.CimArgmaxOp: "argmax_width",
    cd.CimMaxOp: "argmax_width",
    cd.CimReluOp: None,
    cd.CimRsqrtOp: None,
    cd.CimGeluOp: None,
    cd.CimSiluOp: None,
    cd.CimEluOp: None,
    cd.CimLeakyReluOp: None,
    cd.CimLayerNormOp: "ln_lanes",
    # per-token int8 quantisation and the select (kernels/gpt2.py): data width only (the
    # abs-max tree has one characterized size, the per-token rounding one mantissa width)
    nd.CinmRowAbsmaxOp: None,
    nd.CinmQuantizePerTokenOp: None,
    nd.CinmWhereOp: None,
    cd.CimRowAbsmaxOp: None,
    cd.CimQuantizePerTokenOp: None,
    cd.CimVecSelectOp: None,
}

# op type -> its circuit-impl config field (a string): reduce_vec's circuit ("tree",
# or "vec_add" reused over log2(width) halving rounds)
_IMPL_FIELD_OF_OP_TYPE = {
    nd.CinmReduceVecOp: "reduce_impl",
    cd.CimReduceVecOp: "reduce_impl",
}


def reduce_size_of(op) -> int:
    """How many values each reduction of a reduce op sums: its operand's reduced dim
    (cinm.reduceVec always reduces the last)."""
    shape = shape_of(op.operand.type)
    if type(op) is cd.CimReduceVecOp:
        return shape[op.reduce_dim.value.data]
    return shape[-1]


def reduce_tree_inputs(reduce_size: int) -> int:
    """The reduce tree a reduction of reduce_size values uses: the smallest
    characterized tree with at least that many inputs, or the largest (then several
    passes) for a larger reduction."""
    sizes = sorted(load_results.characterized_sizes("reduce_vec"))
    for n in sizes:
        if n >= reduce_size:
            return n
    return sizes[-1]


def stamp_accel_widths(module: ModuleOp, config: dict) -> None:
    for op in module.walk():
        op_type = type(op)
        if op_type not in _LANE_FIELD_OF_OP_TYPE:
            continue
        lane_field = _LANE_FIELD_OF_OP_TYPE[op_type]
        fx_name = fx_name_of(op)
        entry = config.get(fx_name)
        if entry is None:
            raise KeyError(f"{fx_name}: no entry in application config for this {type(op).__name__!r} op")
        impl_field = _IMPL_FIELD_OF_OP_TYPE.get(op_type)
        required = ("data_width",) if lane_field is None else (lane_field, "data_width")
        if impl_field is not None:
            required = required + (impl_field,)
        for field in required:
            if field not in entry:
                raise KeyError(f"{fx_name}: application-config entry (kind {entry.get('kind')!r}) is missing required key {field!r}")
        if lane_field is not None:
            op.attributes[lane_field] = IntegerAttr(int(entry[lane_field]), i64)
        if op_type in _REDUCE_OP_TYPES:
            op.attributes["reduce_width"] = IntegerAttr(reduce_tree_inputs(reduce_size_of(op)), i64)
        if impl_field is not None:
            op.attributes[impl_field] = StringAttr(entry[impl_field])
        op.attributes["accel_data_width"] = IntegerAttr(int(entry["data_width"]), i64)
