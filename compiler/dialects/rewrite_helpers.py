"""Helpers shared by the RewritePattern passes (lowering, tiling, ...): a TensorType's
shape as ints, the scalar-or-operand rhs, and the provenance attributes a replacement
op must carry over from the op it replaces."""

from xdsl.dialects.builtin import TensorType

from compiler.dialects.attr_helpers import float_from_prop

# the attributes copied from a matched op onto its replacement
_PROVENANCE_KEYS = (
    "fx_name",                     # keys the op's config entry
    "logical_matmul_name",
    "logical_matmul_source_op",
    "logical_matmul_source_name",
    # stamped by xdsl_accel_width_stamp.py on the cinm op; xdsl_timing.py reads them
    # off the final cim op
    "reduce_width",
    "argmax_width",
    "accel_data_width",            # the fixed-point width (32: Q12.20, 16: Q8.8)
)


def shape_of(tensor_type: TensorType) -> list[int]:
    return [int(d.data) for d in tensor_type.shape.data]


def dropped_last_dim(tensor_type: TensorType) -> TensorType:
    return TensorType(tensor_type.element_type, shape_of(tensor_type)[:-1])


def scalar_rhs_of(op):
    """A binary_op_scalar_rhs op's rhs: the operand, or else the rhs_scalar float."""
    if op.rhs is not None:
        return op.rhs
    return float_from_prop(op.rhs_scalar)


def copy_provenance(src_op, dst_op) -> None:
    """Copies the _PROVENANCE_KEYS attributes from a matched op onto its replacement."""
    for key in _PROVENANCE_KEYS:
        value = src_op.attributes.get(key)
        if value is not None:
            dst_op.attributes[key] = value


def require_lowered(module, dialect_prefix: str, stage: str) -> None:
    """Raises if any op of the dialect (name prefix, e.g. "cinm.") is left after the
    lowering `stage`: an op with no pattern would otherwise pass silently into the
    next dialect."""
    left = [op for op in module.walk() if op.name.startswith(dialect_prefix)]
    if left:
        names = sorted({
            f"{op.name} ({op.attributes['fx_name'].data})" if "fx_name" in op.attributes else f"{op.name} (no fx_name)"
            for op in left
        })
        raise NotImplementedError(f"{stage}: {len(left)} op(s) have no lowering: {names}")
