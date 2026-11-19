"""group_norm: one function per impl.

- "lut" (the only impl, there is no group-norm circuit): the decomposition into
  mean, centering, variance, sqrt, reciprocal, normalize and the affine step, in cinm
  ops; layer_norm.py's rsqrt patterns then fold the sqrt / reciprocal chain.
"""
from xdsl.dialects.builtin import TensorType
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of
from compiler.ops.dispatch import dispatch, get_entry

KIND = "group_norm"


def _handle_lut_decompose(op, entry: dict, rewriter: PatternRewriter) -> None:
    x = op.operand
    elem_ty = x.type.element_type
    in_shape = shape_of(x.type)
    batch = in_shape[0]
    channels = int_from_prop(op.channels)
    num_groups = int_from_prop(op.num_groups)
    spatial = 1
    for d in in_shape[2:]:
        spatial *= d
    group_size = channels // num_groups
    reduce_len = group_size * spatial
    inv_len = 1.0 / float(reduce_len)
    eps = float_from_prop(op.eps)

    grouped_ty = TensorType(elem_ty, [batch, num_groups, reduce_len])
    reduced_ty = TensorType(elem_ty, [batch, num_groups, 1])

    grouped = nd.CinmReshapeOp(x, [batch, num_groups, reduce_len], grouped_ty)
    mean_sum = nd.CinmReduceVecOp(grouped.result, reduced_ty)
    mean = nd.CinmVecElMulOp(mean_sum.result, inv_len, reduced_ty)
    neg_mean = nd.CinmVecElMulOp(mean.result, -1.0, reduced_ty)
    centered = nd.CinmVecAddOp(grouped.result, neg_mean.result, grouped_ty)
    centered_sq = nd.CinmVecElMulOp(centered.result, centered.result, grouped_ty)
    var_sum = nd.CinmReduceVecOp(centered_sq.result, reduced_ty)
    var = nd.CinmVecElMulOp(var_sum.result, inv_len, reduced_ty)
    # a scalar add built as one: no later pass turns a scalar-rhs vec_add into it
    var_eps = nd.CinmScalarAddOp(var.result, eps, reduced_ty)
    std = nd.CinmSqrtOp(var_eps.result, reduced_ty)
    inv_std = nd.CinmReciprocalOp(std.result, reduced_ty)
    normalized = nd.CinmVecElMulOp(centered.result, inv_std.result, grouped_ty)
    y = nd.CinmReshapeOp(normalized.result, in_shape, op.result.type)

    new_ops = [grouped, mean_sum, mean, neg_mean, centered, centered_sq, var_sum, var, var_eps, std, inv_std, normalized, y]
    for n in new_ops:
        copy_provenance(op, n)

    if op.weight_name is not None:
        affine_shape = [1, channels] + [1] * (len(in_shape) - 2)
        affine_ty = TensorType(elem_ty, affine_shape)
        channel_ty = TensorType(elem_ty, [channels])
        gamma = ConstRefOp(op.weight_name.data, channel_ty)
        gamma_r = nd.CinmReshapeOp(gamma.result, affine_shape, affine_ty)
        y_affine = nd.CinmVecElMulOp(y.result, gamma_r.result, op.result.type)
        copy_provenance(op, y_affine)
        new_ops += [gamma, gamma_r, y_affine]
        y = y_affine
        beta = ConstRefOp(op.bias_name.data, channel_ty)
        beta_r = nd.CinmReshapeOp(beta.result, affine_shape, affine_ty)
        y_biased = nd.CinmVecAddOp(y.result, beta_r.result, op.result.type)
        copy_provenance(op, y_biased)
        new_ops += [beta, beta_r, y_biased]
        y = y_biased

    rewriter.replace_matched_op(new_ops, new_results=y.results)


def build_impl_handlers() -> dict:
    return {"lut": _handle_lut_decompose}


class GroupNormCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmGroupLikeNormOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_group_norm_decompose_canon(module, config: dict) -> None:
    PatternRewriteWalker(GroupNormCanonPattern(config)).rewrite_module(module)
