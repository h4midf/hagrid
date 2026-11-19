"""Gives every value its hardware number format, from the variant config.

Runs once, right after the cinm -> cim lowering (convert_stationary_to_cim_stationary):
every value that exists on the hardware is then in the IR, and the config has been
fully applied. Integer types (the int8 model's ui8 / si8 / i32) already are the
hardware format and stay as they are. Every float value becomes the fixed-point
format of its variant's data_width (cim_dialect.fixed_type_for):

  op with a config entry           its entry's data_width
  shape op (view, transpose, ...)  its input's format
  cat                              its inputs' format (they must agree)
  const.ref / save_stationary      the format its consumers read it at (must agree)
  block argument                   the format its consumers read it at (must agree)

A matvec must have integer operands (the matmuls are int8). A float value with no
rule raises. bit_width(value) is the one width every reader uses.
"""
from xdsl.dialects.builtin import FunctionType, IntegerType, TensorType
from xdsl.dialects.func import FuncOp, ReturnOp
from xdsl.ir import Attribute, Operation, SSAValue
from xdsl.rewriter import Rewriter

from compiler.dialects import cim_dialect as cd
from compiler.dialects.cim_dialect import FixedType, fixed_type_for
from compiler.dialects.common_dialect import CommonReshapeOp, CommonSliceOp, ConstRefOp

_SHAPE_OPS = (
    cd.CimTransposeOp, cd.CimBroadcastOp, cd.CimViewOp, cd.CimPermuteOp, cd.CimReshapeOp,
    cd.CimUnfoldOp, cd.CimContiguousOp, CommonSliceOp, CommonReshapeOp,
)
# ops that only hold or pass a value on: their format is the one their consumers read
_HOLDER_OPS = (ConstRefOp, cd.CimSaveStationaryOp)


def bit_width(value: SSAValue) -> int:
    """Bits per element of a value: its integer or fixed-point element type."""
    element = value.type.element_type
    if isinstance(element, IntegerType):
        return element.width.data
    if isinstance(element, FixedType):
        return element.width.data
    raise TypeError(f"{value} has element type {element}: not a hardware format (run assign_fixed_point_types)")


def _is_hardware_format(element: Attribute) -> bool:
    return isinstance(element, (IntegerType, FixedType))


def _entry_format(op: Operation, config: dict) -> FixedType:
    """The fixed-point format of the op's config entry; raises if it has none."""
    fx_name = op.attributes["fx_name"].data
    return fixed_type_for(int(config[fx_name]["data_width"]))


def _has_entry(op: Operation, config: dict) -> bool:
    return "fx_name" in op.attributes and op.attributes["fx_name"].data in config


def _format_read_by_consumers(value: SSAValue, config: dict) -> FixedType:
    """The format every consumer of `value` reads it at; raises if they disagree or
    a consumer has no rule. Looks through holders and shape ops to the real reader."""
    formats = set()
    for use in value.uses:
        consumer = use.operation
        if isinstance(consumer, ReturnOp):
            continue
        if isinstance(consumer, _HOLDER_OPS) or isinstance(consumer, _SHAPE_OPS):
            formats.add(_format_read_by_consumers(consumer.results[0], config))
        elif _has_entry(consumer, config):
            formats.add(_entry_format(consumer, config))
        else:
            raise NotImplementedError(f"no number-format rule for {value} read by {consumer.name}")
    if len(formats) != 1:
        raise ValueError(f"{value} is read in {len(formats)} formats: {formats}")
    return formats.pop()


def _format_of_result(op: Operation, config: dict) -> FixedType:
    if isinstance(op, _HOLDER_OPS):
        return _format_read_by_consumers(op.results[0], config)
    if isinstance(op, _SHAPE_OPS):
        return op.operands[0].type.element_type
    if isinstance(op, cd.CimCatOp):
        formats = {operand.type.element_type for operand in op.operands}
        if len(formats) != 1:
            raise ValueError(f"cat {op.attributes['fx_name'].data} joins inputs in different formats: {formats}")
        return formats.pop()
    if _has_entry(op, config):
        return _entry_format(op, config)
    raise NotImplementedError(f"no number-format rule for {op.name}")


def _retype(value: SSAValue, element: Attribute) -> None:
    Rewriter.replace_value_with_new_type(value, TensorType(element, value.type.get_shape()))


def assign_fixed_point_types(module, config: dict) -> None:
    func_op = next(op for op in module.body.ops if isinstance(op, FuncOp))
    block = func_op.body.block

    for arg in list(block.args):
        if not _is_hardware_format(arg.type.element_type):
            _retype(arg, _format_read_by_consumers(arg, config))

    for op in list(block.ops):
        if isinstance(op, cd.CimMatvecOp):
            for operand in op.operands:
                if not isinstance(operand.type.element_type, IntegerType):
                    raise TypeError(f"matvec {op.attributes['fx_name'].data} has a {operand.type.element_type} "
                                    "operand: matmuls must be int8 (use a quantized model)")
        for result in list(op.results):
            if not _is_hardware_format(result.type.element_type):
                _retype(result, _format_of_result(op, config))

    return_op = next(op for op in block.ops if isinstance(op, ReturnOp))
    func_op.function_type = FunctionType.from_lists(
        [arg.type for arg in block.args], [operand.type for operand in return_op.operands]
    )
