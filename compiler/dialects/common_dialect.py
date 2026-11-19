"""Ops shared by every stage of the pipeline (gen, cinm, cim)."""

from xdsl.dialects.builtin import ArrayAttr, IntAttr, IntegerType, Signedness, StringAttr
from xdsl.ir import Dialect
from xdsl.irdl import IRDLOperation, irdl_op_definition, operand_def, prop_def, result_def

from compiler.dialects.attr_helpers import int_list_prop

# int8 element types of the quantized model: weights (qint8) are signed,
# activations (quint8) unsigned
si8 = IntegerType(8, Signedness.SIGNED)
ui8 = IntegerType(8, Signedness.UNSIGNED)

# a Python slice's None bound in CommonSliceOp's int lists (an ArrayAttr[IntAttr] cannot
# hold None); larger than any tensor dim
SLICE_NONE = 2**62


@irdl_op_definition
class ConstRefOp(IRDLOperation):
    """A named weight or buffer (an fx get_attr, e.g. "layers.0.Wq.weight"): a reference
    by name, its values are not in the IR."""

    name = "const.ref"

    sym_name = prop_def(StringAttr)
    result = result_def()

    def __init__(self, sym_name: str, result_type):
        super().__init__(
            result_types=[result_type],
            properties={"sym_name": StringAttr(sym_name)},
        )


@irdl_op_definition
class CommonSliceOp(IRDLOperation):
    """tensor[i0:j0:k0, i1:j1:k1, ...] (an fx operator.getitem), carried unchanged from
    gen to cim. starts / stops / steps per dim, SLICE_NONE for a None bound; fewer dims
    than the operand's rank leave the trailing dims whole. Every bound is an int, resolved
    when the bridge builds the op."""

    name = "common.slice"
    operand = operand_def()
    starts = prop_def(ArrayAttr[IntAttr])
    stops = prop_def(ArrayAttr[IntAttr])
    steps = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, starts, stops, steps, result_type):
        super().__init__(
            operands=[operand], result_types=[result_type],
            properties={"starts": int_list_prop(starts), "stops": int_list_prop(stops), "steps": int_list_prop(steps)},
        )


@irdl_op_definition
class CommonReshapeOp(IRDLOperation):
    """The shape fixup after a CommonSliceOp for an int index: x[:, 0, :] is built as
    x[:, 0:1, :] then this reshape, which drops the indexed dim."""

    name = "common.reshape"
    operand = operand_def()
    shape = prop_def(ArrayAttr[IntAttr])
    result = result_def()

    def __init__(self, operand, shape, result_type):
        super().__init__(operands=[operand], result_types=[result_type], properties={"shape": int_list_prop(shape)})


Common = Dialect("const", [ConstRefOp, CommonSliceOp, CommonReshapeOp])
