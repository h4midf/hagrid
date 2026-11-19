"""The `mapped` dialect: the result of hardware mapping on top of `cim` (which pf each
op runs at, when, and on which arrays), kept separate from the computation `cim`
describes.

mapped.schedule wraps a cim op's result (same type) with its schedule:
- pf, start_time / end_time; interval, unit_size and count (xdsl_timing.py's
  OutputGenSpec: the streamed output's pace)
- sync_name: its sync region (annotate_syncs)
- residency: "nonvolatile" (a cim.save_stationary result), "volatile" (an op that
  occupies arrays, MAPPED_OPERATION_OPS) or "passthrough" (no arrays of its own)
- array_ids (volatile ops only): its physical arrays, from interval coloring over the
  peak-count event stream (peak id + 1 = the volatile array count)
- stage: its connected component of the [start_time, end_time) overlap graph (ops in
  different stages never run at the same time)

mapped.stationary_placement records one (stationary value, consumer) edge's arrays and
live window: per edge, since one stationary value can feed several consumers.
"""

from xdsl.dialects.builtin import ArrayAttr, FloatAttr, IntAttr, IntegerAttr, StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, opt_prop_def, prop_def, result_def

from compiler.dialects.attr_helpers import float_prop, int_list_prop, int_prop


@irdl_op_definition
class MappedScheduleOp(IRDLOperation):
    """A cim op's result (operand) with its schedule; result is the same value."""

    name = "mapped.schedule"
    operand = operand_def()
    pf = prop_def(IntegerAttr)
    start_time = prop_def(FloatAttr)
    end_time = prop_def(FloatAttr)
    interval = prop_def(FloatAttr)
    unit_size = prop_def(FloatAttr)
    count = prop_def(IntegerAttr)
    sync_name = opt_prop_def(StringAttr)
    residency = opt_prop_def(StringAttr)
    array_ids = opt_prop_def(ArrayAttr[IntAttr])
    stage = opt_prop_def(IntegerAttr)
    result = result_def()

    def __init__(
        self,
        operand,
        pf: int,
        start_time: float,
        end_time: float,
        interval: float,
        unit_size: float,
        count: int,
        result_type,
        sync_name: str | None = None,
        residency: str | None = None,
        array_ids=None,
        stage: int | None = None,
    ):
        props = {
            "pf": int_prop(pf),
            "start_time": float_prop(start_time),
            "end_time": float_prop(end_time),
            "interval": float_prop(interval),
            "unit_size": float_prop(unit_size),
            "count": int_prop(count),
        }
        if sync_name is not None:
            props["sync_name"] = StringAttr(sync_name)
        if residency is not None:
            props["residency"] = StringAttr(residency)
        if array_ids is not None:
            props["array_ids"] = int_list_prop(array_ids)
        if stage is not None:
            props["stage"] = int_prop(stage)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class MappedStationaryPlacementOp(IRDLOperation):
    """One (stationary value, consumer) edge's arrays and live window; no result."""

    name = "mapped.stationary_placement"
    stationary = operand_def()
    consumer = operand_def()
    array_ids = prop_def(ArrayAttr[IntAttr])
    live_start = prop_def(FloatAttr)
    live_end = prop_def(FloatAttr)

    def __init__(self, stationary, consumer, array_ids, live_start: float, live_end: float):
        super().__init__(
            operands=[stationary, consumer],
            properties={
                "array_ids": int_list_prop(array_ids),
                "live_start": float_prop(live_start),
                "live_end": float_prop(live_end),
            },
        )


Mapped = Dialect("mapped", [MappedScheduleOp, MappedStationaryPlacementOp])
