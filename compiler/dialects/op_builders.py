"""Factories for the op shapes the dialects share: one tensor in, one out; two tensors
in, one out; and a binary op whose rhs, or either side, may be a Python scalar. Ops
with properties (shape, dim, alpha, ...) are defined in their dialect."""

from xdsl.dialects.builtin import FloatAttr, f64
from xdsl.ir import SSAValue
from xdsl.irdl import (
    AttrSizedOperandSegments,
    IRDLOperation,
    irdl_op_definition,
    operand_def,
    opt_operand_def,
    opt_prop_def,
    result_def,
)


def unary_op(op_name: str):
    @irdl_op_definition
    class _Unary(IRDLOperation):
        name = op_name
        operand = operand_def()
        result = result_def()

        def __init__(self, operand, result_type):
            super().__init__(operands=[operand], result_types=[result_type])

    _Unary.__name__ = op_name.replace(".", "_") + "Op"
    return _Unary


def binary_op(op_name: str):
    @irdl_op_definition
    class _Binary(IRDLOperation):
        name = op_name
        lhs = operand_def()
        rhs = operand_def()
        result = result_def()

        def __init__(self, lhs, rhs, result_type):
            super().__init__(operands=[lhs, rhs], result_types=[result_type])

    _Binary.__name__ = op_name.replace(".", "_") + "Op"
    return _Binary


def binary_op_scalar_rhs(op_name: str):
    """lhs op rhs, rhs either a tensor operand or a Python scalar (rhs_scalar), e.g.
    x * 0.125 or var + eps: exactly one of rhs and rhs_scalar is set."""

    @irdl_op_definition
    class _BinaryScalarRhs(IRDLOperation):
        name = op_name
        lhs = operand_def()
        rhs = opt_operand_def()
        rhs_scalar = opt_prop_def(FloatAttr)
        result = result_def()

        def __init__(self, lhs, rhs, result_type):
            if isinstance(rhs, SSAValue):
                super().__init__(operands=[lhs, rhs], result_types=[result_type])
            else:
                super().__init__(
                    operands=[lhs, None],
                    result_types=[result_type],
                    properties={"rhs_scalar": FloatAttr(float(rhs), f64)},
                )

    _BinaryScalarRhs.__name__ = op_name.replace(".", "_") + "Op"
    return _BinaryScalarRhs


def binary_op_either_scalar(op_name: str):
    """Like binary_op_scalar_rhs, but either side may be the scalar (1.0 / std): exactly
    one of lhs / lhs_scalar and one of rhs / rhs_scalar, at least one side a tensor."""

    @irdl_op_definition
    class _BinaryEitherScalar(IRDLOperation):
        name = op_name
        irdl_options = (AttrSizedOperandSegments(),)
        lhs = opt_operand_def()
        lhs_scalar = opt_prop_def(FloatAttr)
        rhs = opt_operand_def()
        rhs_scalar = opt_prop_def(FloatAttr)
        result = result_def()

        def __init__(self, lhs, rhs, result_type):
            operands = []
            props = {}
            if isinstance(lhs, SSAValue):
                operands.append(lhs)
            else:
                operands.append(None)
                props["lhs_scalar"] = FloatAttr(float(lhs), f64)
            if isinstance(rhs, SSAValue):
                operands.append(rhs)
            else:
                operands.append(None)
                props["rhs_scalar"] = FloatAttr(float(rhs), f64)
            super().__init__(operands=operands, result_types=[result_type], properties=props)

    _BinaryEitherScalar.__name__ = op_name.replace(".", "_") + "Op"
    return _BinaryEitherScalar
