"""xDSL properties (int, int list, float, bool) built from Python values and read back,
shared by the dialects, and the checks for the raw forms of an int8 matmul op."""

from xdsl.dialects.builtin import (
    ArrayAttr,
    FloatAttr,
    IntAttr,
    IntegerAttr,
    f64,
    i1,
    i64,
)


def int_prop(value: int) -> IntegerAttr:
    return IntegerAttr(int(value), i64)


def int_list_prop(values) -> ArrayAttr[IntAttr]:
    if not isinstance(values, (list, tuple)):
        values = (values,)
    return ArrayAttr([IntAttr(int(v)) for v in values])


def float_prop(value: float) -> FloatAttr:
    return FloatAttr(float(value), f64)


def bool_prop(value: bool) -> IntegerAttr:
    return IntegerAttr(1 if value else 0, i1)


def int_from_prop(attr: IntegerAttr) -> int:
    return int(attr.value.data)


def int_list_from_prop(attr: ArrayAttr[IntAttr]) -> list[int]:
    return [int(e.data) for e in attr.data]


def float_from_prop(attr: FloatAttr) -> float:
    return float(attr.value.data)


def bool_from_prop(attr: IntegerAttr) -> bool:
    return bool(attr.value.data)


def is_raw_matmul(combined_scale, combined_scale_name, output_scale, output_zero_point, bias) -> bool:
    """True for the raw form of an int8 matmul op (gen.quantized_linear,
    gen.quantized_matmul, cinm.quantized_matmul): no combined scale, output scale,
    output zero point or bias, so the result is the int32 accumulator alone."""
    return all(v is None for v in (combined_scale, combined_scale_name, output_scale, output_zero_point, bias))


def is_group_rescaled_raw(combined_scale, combined_scale_name, output_scale, output_zero_point, bias) -> bool:
    """True for the group-rescaled raw form of a group-wise weight's matmul
    (gen.quantized_linear, cinm.quantized_matmul): only the [groups, out] factors
    (combined_scale_name), so the result is the groups' rescaled sum, not requantized."""
    return combined_scale_name is not None and all(
        v is None for v in (combined_scale, output_scale, output_zero_point, bias))
