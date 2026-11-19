"""The `hcim` dialect: a device-level instruction set for the mapped, scheduled
program, replacing `cim.*` at the materialized stage.

- Channels carry values: `hcim.execute` (HcimChannelOp) computes one `kind` on its
  operands and produces a channel with its logical-tick range (from xdsl_timing.py's
  OutputGenSpec.count, not nanoseconds).
- `hcim.alloc` is the only target-specific op: it requests array units, each result
  reporting the nonvolatile / volatile split the target grants. Everything else is
  target-agnostic.
- Nothing is freed: nonvolatile storage is permanent, and a functional unit
  (`hcim.alloc_fu`, its own always-volatile pool) lives for the range of the
  hcim.execute that consumes it. matvec / lut_search run in the arrays and need no
  functional unit.
"""

from xdsl.dialects.builtin import ArrayAttr, BoolAttr, FloatAttr, IntAttr, StringAttr, TensorType
from xdsl.ir import Attribute, Dialect, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import (
    IRDLOperation,
    irdl_attr_definition,
    irdl_op_definition,
    operand_def,
    opt_prop_def,
    prop_def,
    result_def,
    var_operand_def,
    var_result_def,
)

from compiler.dialects.attr_helpers import bool_prop, float_prop, int_list_prop

# the functional-unit pools (HcimAllocFuOp's kind)
FU_KINDS = ("reduce_tree", "argmax_tree", "vec_add", "vec_el_mul")

# HcimChannelOp's kinds; the ones in _CHANNEL_KIND_TO_FU_KIND need a functional unit
CHANNEL_KINDS = ("matvec", "reduce", "lut_search", "vec_el_mul", "vec_add", "argmax", "max", "cat")

_CHANNEL_KIND_TO_FU_KIND = {
    "reduce": "reduce_tree",
    "argmax": "argmax_tree",
    "max": "argmax_tree",
    "vec_add": "vec_add",
    "vec_el_mul": "vec_el_mul",
}


@irdl_attr_definition
class HcimUnitType(ParametrizedAttribute, TypeAttribute):
    """An array allocation: its nonvolatile and volatile unit counts (both always
    present; a unit is one of the target's arrays)."""

    name = "hcim.unit"

    nonvolatile: IntAttr
    volatile: IntAttr

    def __init__(self, nonvolatile: int, volatile: int):
        super().__init__(IntAttr(nonvolatile), IntAttr(volatile))


@irdl_attr_definition
class HcimFuType(ParametrizedAttribute, TypeAttribute):
    """A claim on `count` units of one functional-unit pool (FU_KINDS); always volatile."""

    name = "hcim.fu"

    kind: StringAttr
    count: IntAttr

    def __init__(self, kind: str, count: int):
        if kind not in FU_KINDS:
            raise ValueError(f"HcimFuType: unknown functional-unit kind {kind!r}, expected one of {FU_KINDS}")
        super().__init__(StringAttr(kind), IntAttr(count))


@irdl_attr_definition
class HcimChannelType(ParametrizedAttribute, TypeAttribute):
    """A live wire carrying `inner` (usually a TensorType). resident: a nonvolatile
    array's connection, always live; otherwise live for its producer's range."""

    name = "hcim.channel"

    inner: Attribute
    resident: IntAttr

    def __init__(self, inner: Attribute, resident: bool = False):
        super().__init__(inner, IntAttr(1 if resident else 0))


@irdl_op_definition
class HcimAllocOp(IRDLOperation):
    """Requests counts[i] array units per result; each result's HcimUnitType is the
    split the target grants (on 1F1C, nonvolatile = volatile: each array is both)."""

    name = "hcim.alloc"

    counts = prop_def(ArrayAttr[IntAttr])
    results_ = var_result_def(HcimUnitType)

    def __init__(self, counts: list[int], result_types: list[HcimUnitType]):
        super().__init__(result_types=[result_types], properties={"counts": int_list_prop(counts)})


@irdl_op_definition
class HcimAssignNonvolatileOp(IRDLOperation):
    """Stores the weight or table named sym_name (as const.ref) in an allocated unit's
    nonvolatile part."""

    name = "hcim.assign_nonvolatile"

    operand = operand_def(HcimUnitType)
    sym_name = prop_def(StringAttr)
    transposed = opt_prop_def(BoolAttr)
    result = result_def(HcimChannelType)

    def __init__(self, operand, sym_name: str, result_type: HcimChannelType, transposed: bool = False):
        props: dict = {"sym_name": StringAttr(sym_name)}
        if transposed:
            props["transposed"] = bool_prop(True)
        super().__init__(operands=[operand], result_types=[result_type], properties=props)


@irdl_op_definition
class HcimStreamInOp(IRDLOperation):
    """A kernel input (a tensor) into the fabric, as a channel."""

    name = "hcim.stream_in"

    operand = operand_def(TensorType)
    result = result_def(HcimChannelType)

    def __init__(self, operand, result_type: HcimChannelType):
        super().__init__(operands=[operand], result_types=[result_type])


@irdl_op_definition
class HcimChannelOp(IRDLOperation):
    """The compute instruction: `kind` (CHANNEL_KINDS) over operands that may be channels
    or plain tensors (a passthrough's result), over the tick range [range_start,
    range_end); the result is a channel. dim (reduce / argmax / max) and alpha / beta
    (lut_search) only for their kinds."""

    name = "hcim.execute"

    operands_ = var_operand_def()
    kind = prop_def(StringAttr)
    range_start = prop_def(IntAttr)
    range_end = prop_def(IntAttr)
    dim = opt_prop_def(IntAttr)
    alpha = opt_prop_def(FloatAttr)
    beta = opt_prop_def(FloatAttr)
    result = result_def(HcimChannelType)

    def __init__(
        self,
        kind: str,
        operands: list,
        range_start: int,
        range_end: int,
        result_type: HcimChannelType,
        dim: int | None = None,
        alpha: float | None = None,
        beta: float | None = None,
    ):
        if kind not in CHANNEL_KINDS:
            raise ValueError(f"HcimChannelOp: unknown kind {kind!r}, expected one of {CHANNEL_KINDS}")
        props: dict = {
            "kind": StringAttr(kind),
            "range_start": IntAttr(range_start),
            "range_end": IntAttr(range_end),
        }
        if dim is not None:
            props["dim"] = IntAttr(dim)
        if alpha is not None:
            props["alpha"] = float_prop(alpha)
        if beta is not None:
            props["beta"] = float_prop(beta)
        super().__init__(operands=[operands], result_types=[result_type], properties=props)

    @staticmethod
    def fu_kind_for(channel_kind: str) -> str | None:
        """The FU_KINDS pool a kind needs, None for the array kinds."""
        return _CHANNEL_KIND_TO_FU_KIND.get(channel_kind)


@irdl_op_definition
class HcimAllocFuOp(IRDLOperation):
    """Claims count units of a functional-unit pool, for the range of the HcimChannelOp
    that consumes it."""

    name = "hcim.alloc_fu"

    kind = prop_def(StringAttr)
    count = prop_def(IntAttr)
    result = result_def(HcimFuType)

    def __init__(self, kind: str, count: int, result_type: HcimFuType):
        if kind not in FU_KINDS:
            raise ValueError(f"HcimAllocFuOp: unknown functional-unit kind {kind!r}, expected one of {FU_KINDS}")
        super().__init__(result_types=[result_type], properties={"kind": StringAttr(kind), "count": IntAttr(count)})


def _passthrough_op(op_name: str):
    """A shape / layout op: no array, functional unit, channel or range."""

    @irdl_op_definition
    class _Passthrough(IRDLOperation):
        name = op_name
        operand = operand_def()
        result = result_def()

        def __init__(self, operand, result_type):
            super().__init__(operands=[operand], result_types=[result_type])

    _Passthrough.__name__ = op_name.replace(".", "_") + "Op"
    return _Passthrough


HcimViewOp = _passthrough_op("hcim.view")
HcimPermuteOp = _passthrough_op("hcim.permute")
HcimReshapeOp = _passthrough_op("hcim.reshape")
HcimBroadcastOp = _passthrough_op("hcim.broadcast")
HcimTransposeOp = _passthrough_op("hcim.transpose")
HcimUnfoldOp = _passthrough_op("hcim.unfold")


@irdl_op_definition
class HcimEmbeddingOp(IRDLOperation):
    """table[index, :]: a lookup with no cost, like a passthrough but with two operands
    (the table itself is stored like any weight)."""

    name = "hcim.embedding"
    table = operand_def()
    index = operand_def()
    result = result_def()

    def __init__(self, table, index, result_type):
        super().__init__(operands=[table, index], result_types=[result_type])


@irdl_op_definition
class HcimReadoutOp(IRDLOperation):
    """A channel out of the fabric, as the tensor the kernel returns."""

    name = "hcim.readout"

    operand = operand_def(HcimChannelType)
    result = result_def(TensorType)

    def __init__(self, operand, result_type):
        super().__init__(operands=[operand], result_types=[result_type])


Hcim = Dialect(
    "hcim",
    [
        HcimAllocOp,
        HcimAssignNonvolatileOp,
        HcimStreamInOp,
        HcimChannelOp,
        HcimAllocFuOp,
        HcimViewOp,
        HcimPermuteOp,
        HcimReshapeOp,
        HcimBroadcastOp,
        HcimTransposeOp,
        HcimUnfoldOp,
        HcimEmbeddingOp,
        HcimReadoutOp,
    ],
    [HcimUnitType, HcimFuType, HcimChannelType],
)
