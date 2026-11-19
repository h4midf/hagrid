"""The per-op structural descriptors xdsl_segments and xdsl_auto_decompose match repeated
pieces of a block by: _real_values (the values with a 1:1 counterpart across repeats)
and _node_desc (op class, result type, pf cap)."""
from xdsl.ir import BlockArgument, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.xdsl_timing import _defining_op


def _real_values(mapper) -> list:
    """mapper.node_order without the values that have no 1:1 counterpart across
    repeats:
    - BlockArguments: a piece's input is the previous op's output in the full model.
    - ConstRefOp / CimSaveStationaryOp: an identical pair shared by repeats is folded
      into one value. Their pf is always 1, so dropping them loses nothing."""
    result = []
    for v in mapper.node_order:
        if isinstance(v, BlockArgument):
            continue
        op = _defining_op(v)
        if op is None or isinstance(op, (ConstRefOp, cd.CimSaveStationaryOp)):
            continue
        result.append(v)
    return result


def _node_desc(mapper, value: SSAValue) -> tuple:
    op = _defining_op(value)
    return (type(op).__name__, str(value.type), int(mapper.pf_cap_for_value(value)))
