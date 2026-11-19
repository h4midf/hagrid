"""rms_norm: one function per impl.

- "exact": the op stays, lowered to cim.rms_norm and costed on custom_op_library/
  rms_norm (mean square, sqrt and reciprocal through the rsqrt primitive, scale).
- "lut": the decomposition x * 1 / (sqrt(mean(x^2)) + eps) (eps after the sqrt), in
  cinm ops carrying the op's fx_name; layer_norm.py's GeneralReciprocalCanonPattern
  then makes its reciprocal a LUT lookup.
"""
from xdsl.dialects.builtin import TensorType
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, op_type_rewrite_pattern, RewritePattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of
from compiler.ops.dispatch import dispatch, get_entry

KIND = "rms_norm"


def _handle_exact(op, entry: dict, rewriter: PatternRewriter) -> None:
    return


def _handle_lut_decompose(op, entry: dict, rewriter: PatternRewriter) -> None:
    x = op.operand
    hidden = int_from_prop(op.hidden_size)
    inv_hidden = 1.0 / float(hidden)
    eps = float_from_prop(op.eps)
    elem_ty = x.type.element_type
    reduced_ty = TensorType(elem_ty, shape_of(x.type)[:-1] + [1])

    sq = nd.CinmVecElMulOp(x, x, x.type)
    sum_sq = nd.CinmReduceVecOp(sq.result, reduced_ty)
    ms = nd.CinmVecElMulOp(sum_sq.result, inv_hidden, reduced_ty)
    sqrt_ms = nd.CinmSqrtOp(ms.result, reduced_ty)
    # a scalar add built as one (no later pass turns a scalar-rhs vec_add into it)
    s_eps = nd.CinmScalarAddOp(sqrt_ms.result, eps, reduced_ty)
    inv = nd.CinmReciprocalOp(s_eps.result, reduced_ty)

    new_ops = [sq, sum_sq, ms, sqrt_ms, s_eps, inv]
    for n in new_ops:
        copy_provenance(op, n)

    norm = nd.CinmVecElMulOp(x, inv.result, x.type)
    copy_provenance(op, norm)
    new_ops.append(norm)
    y_op = norm

    if op.weight_name is not None:
        gamma_ty = TensorType(elem_ty, [hidden])
        gamma = ConstRefOp(op.weight_name.data, gamma_ty)
        y_affine = nd.CinmVecElMulOp(y_op.result, gamma.result, x.type)
        copy_provenance(op, y_affine)
        new_ops += [gamma, y_affine]
        y_op = y_affine

    rewriter.replace_matched_op(new_ops, new_results=y_op.results)


def build_decompose_impl_handlers() -> dict:
    return {"lut": _handle_lut_decompose, "exact": _handle_exact}


class RMSNormCanonPattern(RewritePattern):
    """The whole CinmRMSNormOp, by impl. Runs before GeneralReciprocalCanonPattern, which
    folds the reciprocal its "lut" decomposition builds."""

    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_decompose_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmRMSNormOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_rms_norm_decompose_canon(module, config: dict) -> None:
    PatternRewriteWalker(RMSNormCanonPattern(config)).rewrite_module(module)
