"""softmax: one function per impl.

- "lut": w = LUT_exp(x), s = reduce_vec(w), r = LUT_recip(s), y = w * r (a scalar
  multiply feeding x is folded into the exp lookup).
- a circuit (VARIANT_IMPLS): the op stays whole, with the impl stamped as its "variant"
  attribute; it lowers to cim.softmax, whose variant picks the costed circuit:
    "exact"    custom_op_library/softmax/rtl/softmax_row8.v
    "sole"     custom_op_library/softmax/rtl/e2softmax_row8.v (SOLE, ICCAD 2023)
    "quark"    custom_op_library/quark_shared/rtl/quark_softmax_row8.v (QUARK, ICCAD 2025)
    "gqa_lut"  custom_op_library/softmax/rtl/gqa_lut_softmax_row8.v (GQA-LUT, DAC 2024)
    "ibert"    custom_op_library/softmax/rtl/ibert_softmax_row8.v (I-BERT, ICML 2021)
"""
from xdsl.dialects.builtin import StringAttr, TensorType, f32
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of
from compiler.ops.dispatch import dispatch, fx_name_of, get_entry, require

KIND = "softmax"


class _LutHandler:
    """impl "lut": one exp table and one reciprocal table, shared by the matches of one
    pattern instance. Known bug: the reciprocal table is built for the first match's
    row length d and reused for every softmax, also one with another d."""

    def __init__(self):
        self._exp_table: ConstRefOp | None = None
        self._rcp_table: ConstRefOp | None = None

    def __call__(self, op, entry: dict, rewriter: PatternRewriter) -> None:
        fx_name = fx_name_of(op)
        xmin = require(entry, "xmin", fx_name)
        xmax = require(entry, "xmax", fx_name)
        nbins = require(entry, "nbins", fx_name)
        rcp_nbins = require(entry, "rcp_nbins", fx_name)

        d = shape_of(op.operand.type)[-1]

        x_for_exp = op.operand
        input_scale = 1.0
        folded_scale_op = None
        producer = op.operand.owner
        if isinstance(producer, nd.CinmVecElMulOp) and producer.rhs_scalar is not None:
            x_for_exp = producer.lhs
            input_scale = float_from_prop(producer.rhs_scalar)
            folded_scale_op = producer

        alpha_exp = float((nbins - 1) / (xmax - xmin))
        beta_exp = float(-xmin * alpha_exp)
        alpha_rcp = float((rcp_nbins - 1) / (d - 1.0))
        beta_rcp = float(-1.0 * alpha_rcp)

        new_ops = []
        if self._exp_table is None:
            self._exp_table = ConstRefOp(f"softmax_exp_lut_n{nbins}", TensorType(f32, [nbins]))
            new_ops.append(self._exp_table)
        w_type = x_for_exp.type
        w = nd.CinmLUTOp(self._exp_table.result, x_for_exp, w_type, alpha=alpha_exp * input_scale, beta=beta_exp)
        new_ops.append(w)
        copy_provenance(op, w)

        s_type = TensorType(w_type.element_type, shape_of(w_type)[:-1] + [1])
        s = nd.CinmReduceVecOp(w.result, s_type)
        new_ops.append(s)
        copy_provenance(op, s)

        if self._rcp_table is None:
            self._rcp_table = ConstRefOp(f"softmax_rcp_lut_n{rcp_nbins}_d{d}", TensorType(f32, [rcp_nbins]))
            new_ops.append(self._rcp_table)
        r = nd.CinmLUTOp(self._rcp_table.result, s.result, s_type, alpha=alpha_rcp, beta=beta_rcp)
        new_ops.append(r)
        copy_provenance(op, r)

        y = nd.CinmVecElMulOp(w.result, r.result, op.result.type)
        new_ops.append(y)
        copy_provenance(op, y)

        rewriter.replace_matched_op(new_ops, new_results=y.results)
        if folded_scale_op is not None and not folded_scale_op.results[0].uses:
            rewriter.erase_op(folded_scale_op)


VARIANT_IMPLS = ("exact", "sole", "quark", "gqa_lut", "ibert")


def _handle_variant(op, entry: dict, rewriter: PatternRewriter) -> None:
    """A circuit impl: the op stays, its impl stamped as "variant"."""
    fx_name = fx_name_of(op)
    variant = require(entry, "impl", fx_name)
    op.attributes["variant"] = StringAttr(variant)


def build_impl_handlers() -> dict:
    """New handlers per pattern instance (the "lut" tables live as long as it does)."""
    return {"lut": _LutHandler(), **{v: _handle_variant for v in VARIANT_IMPLS}}


class SoftmaxCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmSoftmaxOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_softmax_canon(module, config: dict) -> None:
    PatternRewriteWalker(SoftmaxCanonPattern(config)).rewrite_module(module)
