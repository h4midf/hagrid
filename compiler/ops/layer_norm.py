"""layer_norm: one function per impl.

- a circuit (VARIANT_IMPLS): a streamed circuit taking the row ln_lanes elements per
  cycle (custom_op_library/layer_norm/characterize_stream.py). The op stays, with the
  impl stamped as its "variant" attribute; it lowers to cim.layer_norm, costed by
  variant, row length, width and lanes. A configuration the circuit is not
  characterized at raises.
    "exact"  custom_op_library/layer_norm/rtl/layer_norm_stream.v (rsqrt by a seed
             table and one Newton step)
    "sole"   custom_op_library/layer_norm/rtl/ailayernorm_stream.v (SOLE, ICCAD 2023)
- "lut": the decomposition into mean, centering, variance, sqrt, reciprocal, normalize
  and the affine step, in cinm ops. Then RsqrtChainCanonPattern folds
  reciprocal(sqrt(var + eps)) (or scalar_div(1.0, sqrt(...))) into one rsqrt LUT on
  var + eps, and GeneralReciprocalCanonPattern makes any other reciprocal(x) a LUT
  lookup at x * alpha + beta.
- "rsqrt": the same decomposition, with reciprocal(sqrt(var + eps)) folded into one
  cinm.rsqrt (the CMOS rsqrt circuit, custom_op_library/rsqrt) instead of a LUT.

The two reciprocal patterns match on the IR shape, so they also fold group_norm's and
rms_norm's decompositions. DivAsVecMulPattern and FoldReciprocalIntoLutPattern (no
config) run between them, from the top-level canon.
"""
from xdsl.dialects.builtin import StringAttr, TensorType, f32
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, op_type_rewrite_pattern, RewritePattern
from xdsl.rewriter import InsertPoint

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of
from compiler.ops._activation_lut_common import fnum
from compiler.ops.dispatch import dispatch, fx_name_of, get_entry, require

KIND = "layer_norm"


# the streamed circuits (custom_op_library/layer_norm/characterize_stream.py)
VARIANT_IMPLS = ("exact", "sole")


def _handle_variant(op, entry: dict, rewriter: PatternRewriter) -> None:
    fx_name = fx_name_of(op)
    variant = require(entry, "impl", fx_name)
    op.attributes["variant"] = StringAttr(variant)


def _handle_lut_decompose(op, entry: dict, rewriter: PatternRewriter) -> None:
    x = op.operand
    hidden = int_from_prop(op.hidden_size)
    inv_hidden = 1.0 / float(hidden)
    eps = float_from_prop(op.eps)
    elem_ty = x.type.element_type
    reduced_ty = TensorType(elem_ty, shape_of(x.type)[:-1] + [1])

    mean_sum = nd.CinmReduceVecOp(x, reduced_ty)
    mean = nd.CinmVecElMulOp(mean_sum.result, inv_hidden, reduced_ty)
    neg_mean = nd.CinmVecElMulOp(mean.result, -1.0, reduced_ty)
    centered = nd.CinmVecAddOp(x, neg_mean.result, x.type)
    centered_sq = nd.CinmVecElMulOp(centered.result, centered.result, x.type)
    var_sum = nd.CinmReduceVecOp(centered_sq.result, reduced_ty)
    var = nd.CinmVecElMulOp(var_sum.result, inv_hidden, reduced_ty)
    # a scalar add built as one (no later pass turns a scalar-rhs vec_add into it):
    # _extract_rsqrt_chain looks for a CinmScalarAddOp
    var_eps = nd.CinmScalarAddOp(var.result, eps, reduced_ty)
    std = nd.CinmSqrtOp(var_eps.result, reduced_ty)
    inv_std = nd.CinmReciprocalOp(std.result, reduced_ty)
    y_op = nd.CinmVecElMulOp(centered.result, inv_std.result, x.type)

    new_ops = [mean_sum, mean, neg_mean, centered, centered_sq, var_sum, var, var_eps, std, inv_std, y_op]
    # every op carries the layer norm's fx_name: the reciprocal patterns match the
    # sqrt / reciprocal ops in the middle and look up the config entry by it
    for n in new_ops:
        copy_provenance(op, n)

    if op.weight_name is not None:
        gamma_ty = TensorType(elem_ty, [hidden])
        gamma = ConstRefOp(op.weight_name.data, gamma_ty)
        y_affine = nd.CinmVecElMulOp(y_op.result, gamma.result, x.type)
        copy_provenance(op, y_affine)
        new_ops += [gamma, y_affine]
        y_op = y_affine
        if op.bias_name is not None:
            beta = ConstRefOp(op.bias_name.data, gamma_ty)
            y_biased = nd.CinmVecAddOp(y_op.result, beta.result, x.type)
            copy_provenance(op, y_biased)
            new_ops += [beta, y_biased]
            y_op = y_biased

    rewriter.replace_matched_op(new_ops, new_results=y_op.results)


def build_decompose_impl_handlers() -> dict:
    # "rsqrt" decomposes the same way as "lut"; only its sqrt/reciprocal sub-chain is
    # folded differently (RsqrtChainCanonPattern: one CMOS rsqrt instead of a LUT)
    return {"lut": _handle_lut_decompose, "rsqrt": _handle_lut_decompose,
            **{v: _handle_variant for v in VARIANT_IMPLS}}


class LayerNormCanonPattern(RewritePattern):
    """The whole CinmLayerNormOp, by impl. Runs before the reciprocal patterns, which
    fold the chain its "lut" / "rsqrt" decomposition builds."""

    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_decompose_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmLayerNormOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_layer_norm_decompose_canon(module, config: dict) -> None:
    PatternRewriteWalker(LayerNormCanonPattern(config)).rewrite_module(module)


def _extract_rsqrt_chain(op):
    """reciprocal(sqrt(scalar_add(...))) or scalar_div(1.0, sqrt(scalar_add(...))).
    Returns (sqrt_op, scalar_add_op) or (None, None)."""
    if isinstance(op, nd.CinmReciprocalOp):
        sqrt_op = op.operand.owner
    elif isinstance(op, nd.CinmScalarDivOp):
        if op.lhs_scalar is None or float_from_prop(op.lhs_scalar) != 1.0:
            return None, None
        if op.rhs is None:
            return None, None
        sqrt_op = op.rhs.owner
    else:
        return None, None
    if not isinstance(sqrt_op, nd.CinmSqrtOp):
        return None, None
    scalar_add_op = sqrt_op.operand.owner
    if not isinstance(scalar_add_op, nd.CinmScalarAddOp):
        return None, None
    return sqrt_op, scalar_add_op


class _RsqrtChainLutHandler:
    """impl "lut": one rsqrt table, shared by the matches of one pattern instance."""

    def __init__(self):
        self._table: ConstRefOp | None = None

    def __call__(self, op, entry: dict, rewriter: PatternRewriter, sqrt_op, scalar_add_op) -> None:
        fx_name = fx_name_of(op)
        nbins = require(entry, "nbins", fx_name)
        xmin = require(entry, "xmin", fx_name)
        xmax = require(entry, "xmax", fx_name)
        if self._table is None:
            self._table = ConstRefOp(f"rsqrt_lut_n{nbins}_x{fnum(xmin)}_X{fnum(xmax)}", TensorType(f32, [nbins]))
            rewriter.insert_op(self._table, InsertPoint.before(scalar_add_op))
        alpha = float((nbins - 1) / (xmax - xmin))
        beta = float(-xmin * alpha)
        new_op = nd.CinmLUTOp(self._table.result, scalar_add_op.result, op.result.type, alpha=alpha, beta=beta)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)
        if not sqrt_op.results[0].uses:
            rewriter.erase_op(sqrt_op)


def _handle_general_reciprocal_lut(op, entry: dict, rewriter: PatternRewriter) -> None:
    fx_name = fx_name_of(op)
    nbins = require(entry, "nbins", fx_name)
    xmin = require(entry, "xmin", fx_name)
    xmax = require(entry, "xmax", fx_name)
    alpha = float((nbins - 1) / (xmax - xmin))
    beta = float(-xmin * alpha)
    table = ConstRefOp(f"recip_lut_n{nbins}_x{fnum(xmin)}_X{fnum(xmax)}", TensorType(f32, [nbins]))
    operand_type = op.operand.type
    mul = nd.CinmVecElMulOp(op.operand, alpha, operand_type)
    copy_provenance(op, mul)
    idx = nd.CinmVecAddOp(mul.result, beta, operand_type)
    copy_provenance(op, idx)
    # the index map is applied by the mul/add above
    new_op = nd.CinmLUTOp(table.result, idx.result, op.result.type, alpha=1.0, beta=0.0)
    copy_provenance(op, new_op)
    rewriter.replace_matched_op([table, mul, idx, new_op], new_results=new_op.results)


def _handle_rsqrt_chain_circuit(op, entry: dict, rewriter: PatternRewriter, sqrt_op, scalar_add_op) -> None:
    """impl "rsqrt": reciprocal(sqrt(var + eps)) becomes one cinm.rsqrt of var + eps,
    computed by the CMOS rsqrt circuit (custom_op_library/rsqrt) like the other
    pointwise ops."""
    new_op = nd.CinmRsqrtOp(scalar_add_op.result, op.result.type)
    copy_provenance(op, new_op)
    rewriter.replace_matched_op(new_op)
    if not sqrt_op.results[0].uses:
        rewriter.erase_op(sqrt_op)


def build_rsqrt_chain_handlers() -> dict:
    return {"lut": _RsqrtChainLutHandler(), "rsqrt": _handle_rsqrt_chain_circuit}


def build_general_reciprocal_handlers() -> dict:
    return {"lut": _handle_general_reciprocal_lut}


def _dispatch_with_extra(op, entry: dict, rewriter, impl_handlers: dict, *extra) -> None:
    """dispatch.dispatch for handlers that take extra args (the rsqrt chain's sqrt_op,
    scalar_add_op)."""
    fx_name = fx_name_of(op)
    impl = require(entry, "impl", fx_name)
    handler = impl_handlers.get(impl)
    if handler is None:
        known = sorted(impl_handlers.keys())
        raise NotImplementedError(
            f"{fx_name}: impl {impl!r} requested, but this operation only has handlers for {known}"
        )
    handler(op, entry, rewriter, *extra)


_RSQRT_CHAIN_KINDS = ("layer_norm", "group_norm", "rms_norm")


def _get_rsqrt_chain_entry(config: dict, op) -> dict:
    """dispatch.get_entry for any of _RSQRT_CHAIN_KINDS: layer_norm's, group_norm's and
    rms_norm's decompositions all build the reciprocal these patterns match (rms_norm
    adds eps after the sqrt, so only GeneralReciprocalCanonPattern matches it)."""
    fx_name = fx_name_of(op)
    entry = config.get(fx_name)
    if entry is None:
        raise KeyError(f"{fx_name}: no entry in application config for this rsqrt-chain op")
    if entry.get("kind") not in _RSQRT_CHAIN_KINDS:
        raise ValueError(
            f"{fx_name}: config entry kind {entry.get('kind')!r} does not match this op's own kind "
            f"(expected one of {_RSQRT_CHAIN_KINDS})"
        )
    return entry


class RsqrtChainCanonPattern(RewritePattern):
    """reciprocal(sqrt(var + eps)) or scalar_div(1.0, sqrt(var + eps)), by impl."""

    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_rsqrt_chain_handlers()

    def match_and_rewrite(self, op, rewriter: PatternRewriter) -> None:
        sqrt_op, scalar_add_op = _extract_rsqrt_chain(op)
        if sqrt_op is None:
            return
        entry = _get_rsqrt_chain_entry(self.config, op)
        _dispatch_with_extra(op, entry, rewriter, self.impl_handlers, sqrt_op, scalar_add_op)


class GeneralReciprocalCanonPattern(RewritePattern):
    """Any other reciprocal(x), except reciprocal(LUTOp) (FoldReciprocalIntoLutPattern's)."""

    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_general_reciprocal_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmReciprocalOp, rewriter: PatternRewriter) -> None:
        if isinstance(op.operand.owner, nd.CinmLUTOp):
            return
        entry = _get_rsqrt_chain_entry(self.config, op)
        _dispatch_with_extra(op, entry, rewriter, self.impl_handlers)
