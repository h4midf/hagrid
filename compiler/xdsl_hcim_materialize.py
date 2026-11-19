"""Writes a converged (TimingOnlyMapping, HardwareMapping) state as an hcim.* program
(dialects/hcim_dialect.py), in a new module: hcim replaces
cim, the cim block is not changed.

Simplifications (open questions in the design doc):
- an HcimChannelOp's range is its own [0, count) (xdsl_timing's OutputGenSpec.count),
  not on a clock shared by the ops;
- a functional unit claim's count is always 1.

Op coverage: matvec, reduce_vec, lut_search(_vector), vec_el_mul, vec_add, argmax, max
and cat become HcimChannelOps; the shape ops and embedding become passthroughs. Any
other op (e.g. a softmax or quantize circuit) raises.
"""

from __future__ import annotations

from xdsl.dialects.builtin import ModuleOp
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Block, Region, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.common_dialect import CommonReshapeOp, CommonSliceOp, ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance
from compiler.dialects.hcim_dialect import (
    HcimAllocFuOp,
    HcimAllocOp,
    HcimAssignNonvolatileOp,
    HcimBroadcastOp,
    HcimChannelOp,
    HcimChannelType,
    HcimEmbeddingOp,
    HcimFuType,
    HcimPermuteOp,
    HcimReadoutOp,
    HcimReshapeOp,
    HcimStreamInOp,
    HcimTransposeOp,
    HcimUnfoldOp,
    HcimUnitType,
    HcimViewOp,
)
from compiler.xdsl_hardware import HardwareMapping
from compiler.xdsl_timing import TimingOnlyMapping, _defining_op, _is_return

_PASSTHROUGH_OP_MAP = {
    cd.CimTransposeOp: HcimTransposeOp,
    cd.CimBroadcastOp: HcimBroadcastOp,
    cd.CimViewOp: HcimViewOp,
    cd.CimPermuteOp: HcimPermuteOp,
    cd.CimReshapeOp: HcimReshapeOp,
    cd.CimUnfoldOp: HcimUnfoldOp,
    CommonSliceOp: HcimViewOp,
    CommonReshapeOp: HcimReshapeOp,
}

_CIM_TO_CHANNEL_KIND = {
    cd.CimMatvecOp: "matvec",
    cd.CimReduceVecOp: "reduce",
    cd.CimLutSearchOp: "lut_search",
    cd.CimLutSearchVectorOp: "lut_search",
    cd.CimVecElMulOp: "vec_el_mul",
    cd.CimVecAddOp: "vec_add",
    cd.CimArgmaxOp: "argmax",
    cd.CimMaxOp: "max",
    cd.CimCatOp: "cat",
}


def _weight_symbol(save_stationary_op: cd.CimSaveStationaryOp) -> tuple[str, bool]:
    """(the const.ref name, transposed) of a save_stationary, whose operand must be a
    const.ref (a transpose is its `transposed` property); raises otherwise."""
    const_op = _defining_op(save_stationary_op.operand)
    if not isinstance(const_op, ConstRefOp):
        raise NotImplementedError(
            f"materialize_hcim: save_stationary operand is {const_op}, not a direct const.ref -- "
            "unhandled real graph shape"
        )
    transposed = save_stationary_op.transposed is not None and bool(save_stationary_op.transposed.value.data)
    return const_op.sym_name.data, transposed


def materialize_hcim(block: Block, mapper: TimingOnlyMapping, hardware: HardwareMapping) -> ModuleOp:
    """A new module with the hcim program of `block`, from mapper / hardware's converged
    state over it; `block` is not changed."""

    new_block = Block(arg_types=[arg.type for arg in block.args])
    value_map: dict[SSAValue, SSAValue] = {}

    # -- pass 1: stationary weights -- one HcimAllocOp for all of them,
    # then one HcimAssignNonvolatileOp each, in real graph order --
    stationary_ops = [op for op in block.ops if isinstance(op, cd.CimSaveStationaryOp)]
    counts = [int(hardware.stationary_arrays_per_copy(op.results[0])) for op in stationary_ops]
    unit_types = [HcimUnitType(c, c) for c in counts]
    if stationary_ops:
        alloc = HcimAllocOp(counts, unit_types)
        new_block.add_op(alloc)
        for stat_op, unit_result, unit_type in zip(stationary_ops, alloc.results, unit_types):
            sym_name, transposed = _weight_symbol(stat_op)
            ch_type = HcimChannelType(unit_type, resident=True)
            assign = HcimAssignNonvolatileOp(unit_result, sym_name, ch_type, transposed=transposed)
            copy_provenance(stat_op, assign)
            new_block.add_op(assign)
            value_map[stat_op.results[0]] = assign.results[0]

    # -- pass 2: block arguments enter the fabric once --
    for old_arg, new_arg in zip(block.args, new_block.args):
        stream_in = HcimStreamInOp(new_arg, HcimChannelType(new_arg.type))
        new_block.add_op(stream_in)
        value_map[old_arg] = stream_in.results[0]

    # -- pass 3: everything else, in mapper.node_order (topological) --
    for value in mapper.node_order:
        if value in value_map:
            continue  # already handled: a stationary result, or (impossible here) a block arg
        op = _defining_op(value)
        if op is None:
            continue  # block arg, already handled above
        if isinstance(op, ConstRefOp):
            # a const.ref is only read through its save_stationary (pass 1)
            if all(isinstance(use.operation, cd.CimSaveStationaryOp) for use in value.uses):
                continue
            raise NotImplementedError(
                f"materialize_hcim: const.ref {op.sym_name.data!r} has a non-save_stationary use -- "
                "unhandled real graph shape"
            )

        passthrough_cls = _PASSTHROUGH_OP_MAP.get(type(op))
        if passthrough_cls is not None:
            new_operand = value_map[op.operand]
            new_op = passthrough_cls(new_operand, value.type)
            copy_provenance(op, new_op)
            new_block.add_op(new_op)
            value_map[value] = new_op.results[0]
            continue

        if isinstance(op, cd.CimEmbeddingOp):
            # a passthrough with two operands (table, index), so not in _PASSTHROUGH_OP_MAP
            new_op = HcimEmbeddingOp(value_map[op.lhs], value_map[op.rhs], value.type)
            copy_provenance(op, new_op)
            new_block.add_op(new_op)
            value_map[value] = new_op.results[0]
            continue

        channel_kind = _CIM_TO_CHANNEL_KIND.get(type(op))
        if channel_kind is None:
            raise NotImplementedError(f"materialize_hcim: no hcim mapping for op {op}")

        spec = mapper.specs[mapper.node_index[value]]
        operands = [value_map[operand] for operand in op.operands]

        fu_kind = HcimChannelOp.fu_kind_for(channel_kind)
        if fu_kind is not None:
            fu = HcimAllocFuOp(fu_kind, 1, HcimFuType(fu_kind, 1))
            new_block.add_op(fu)
            operands = operands + [fu.results[0]]

        dim = None
        if isinstance(op, cd.CimReduceVecOp):
            dim = int(op.reduce_dim.value.data)
        elif isinstance(op, (cd.CimArgmaxOp, cd.CimMaxOp)) and op.reduce_dim is not None:
            dim = int(op.reduce_dim.value.data)

        alpha = beta = None
        if isinstance(op, cd.CimLutSearchOp):
            alpha = float(op.alpha.value.data)
            beta = float(op.beta.value.data)

        channel_op = HcimChannelOp(
            channel_kind, operands, 0, int(spec.count), HcimChannelType(value.type),
            dim=dim, alpha=alpha, beta=beta,
        )
        copy_provenance(op, channel_op)
        new_block.add_op(channel_op)
        value_map[value] = channel_op.results[0]

    # -- pass 4: the sink leaves the fabric: a channel is read out; a passthrough's
    # result is already a tensor --
    return_op = next(op for op in block.ops if _is_return(op))
    sink = return_op.operands[0]
    sink_value = value_map[sink]
    if isinstance(sink_value.type, HcimChannelType):
        readout = HcimReadoutOp(sink_value, sink.type)
        new_block.add_op(readout)
        sink_value = readout.results[0]
    new_block.add_op(ReturnOp(sink_value))

    func_op = FuncOp("main", (list(new_block.arg_types), [sink.type]), Region([new_block]))
    old_func_op = block.parent_op()
    arg_names_attr = old_func_op.attributes.get("arg_names") if old_func_op is not None else None
    if arg_names_attr is not None:
        func_op.attributes["arg_names"] = arg_names_attr
    return ModuleOp([func_op])
