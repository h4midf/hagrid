"""The "lut" impl shared by gelu / relu / elu / leaky_relu: a cinm.LUTOp on an nbins
table over [xmin, xmax] (alpha = (nbins - 1) / (xmax - xmin), beta = -xmin * alpha),
one table per activation and parameters, shared by its call sites."""
from xdsl.dialects.builtin import TensorType, f32
from xdsl.pattern_rewriter import PatternRewriter
from xdsl.rewriter import InsertPoint

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance
from compiler.ops.dispatch import fx_name_of, require


def fnum(value: float) -> str:
    """A number in a table name (-6.0 -> "-6p0")."""
    return str(value).replace(".", "p")


def no_suffix(op) -> str:
    """The table-name suffix of an activation with no parameter (gelu, relu)."""
    return ""


class FusedLutActivationHandler:
    """attr_prefix names the activation ("gelu"/"relu"/"elu"/"leaky_relu");
    extra_suffix(op) -> str bakes the op's own parameter (elu's alpha, leaky_relu's
    negative_slope) into the table name, or is no_suffix."""

    def __init__(self, attr_prefix: str, extra_suffix):
        self.attr_prefix = attr_prefix
        self.extra_suffix = extra_suffix
        self._table_cache: dict[str, ConstRefOp] = {}

    def __call__(self, op, entry: dict, rewriter: PatternRewriter) -> None:
        fx_name = fx_name_of(op)
        xmin = require(entry, "xmin", fx_name)
        xmax = require(entry, "xmax", fx_name)
        nbins = require(entry, "nbins", fx_name)
        suffix = self.extra_suffix(op)
        attr_name = f"{self.attr_prefix}_lut_n{nbins}_x{fnum(xmin)}_X{fnum(xmax)}{suffix}"
        table_op = self._table_cache.get(attr_name)
        if table_op is None:
            table_op = ConstRefOp(attr_name, TensorType(f32, [nbins]))
            self._table_cache[attr_name] = table_op
            rewriter.insert_op(table_op, InsertPoint.before(op))
        alpha = float((nbins - 1) / (xmax - xmin))
        beta = float(-xmin * alpha)
        new_op = nd.CinmLUTOp(table_op.result, op.operand, op.result.type, alpha=alpha, beta=beta)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)
