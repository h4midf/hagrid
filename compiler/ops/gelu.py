"""gelu: one function per impl.

- "lut": the shared activation LUT (_activation_lut_common.py).
- a circuit (VARIANT_IMPLS): the op stays, with the impl stamped as its "variant"
  attribute; the cim lowering makes it cim.gelu's variant, which picks the costed
  circuit:
    "exact"      custom_op_library/gelu/rtl/gelu.v
    "quark"      custom_op_library/quark_shared/rtl/quark_gelu.v (QUARK, ICCAD 2025)
    "wang_tcad"  custom_op_library/gelu/rtl/wang_tcad_gelu24.v (Wang et al., TCAD 2025)
    "gqa_lut"    custom_op_library/gelu/rtl/gqa_lut_gelu8.v (GQA-LUT, DAC 2024)
    "ibert"      custom_op_library/gelu/rtl/ibert_gelu.v (I-BERT, ICML 2021; Q8.8 only)

The GELU form: the frontend records torch's `approximate` on the op ("none", the erf
form, or "tanh", GPT-2's gelu_new). The circuits above fit the erf form. A tanh-form
node takes impl "exact" only, costed on variant "exact_tanh"
(custom_op_library/gelu/rtl/gelu_tanh.v); any other impl on it raises.
"""
from xdsl.dialects.builtin import StringAttr
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.ops._activation_lut_common import FusedLutActivationHandler, no_suffix
from compiler.ops.dispatch import dispatch, fx_name_of, get_entry, require

KIND = "gelu"

VARIANT_IMPLS = ("exact", "quark", "wang_tcad", "gqa_lut", "ibert")
# impls a tanh-form node may take -> the circuit variant costed
TANH_FORM_VARIANTS = {"exact": "exact_tanh"}


def gelu_form(op) -> str:
    """The op's GELU form, "none" (erf) or "tanh" (set by the frontend)."""
    return op.attributes["approximate"].data


def _handle_variant(op, entry: dict, rewriter: PatternRewriter) -> None:
    fx_name = fx_name_of(op)
    variant = require(entry, "impl", fx_name)
    if gelu_form(op) == "tanh":
        variant = TANH_FORM_VARIANTS[variant]
    op.attributes["variant"] = StringAttr(variant)


def build_impl_handlers() -> dict:
    return {"lut": FusedLutActivationHandler("gelu", no_suffix), **{v: _handle_variant for v in VARIANT_IMPLS}}


class GeluCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmGeluOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        if gelu_form(op) == "tanh" and entry.get("impl") not in TANH_FORM_VARIANTS:
            raise NotImplementedError(
                f"{fx_name_of(op)}: tanh-form GELU with impl={entry.get('impl')!r}: only "
                f"{tuple(TANH_FORM_VARIANTS)} has a tanh-form circuit (the others fit the erf form)")
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_gelu_canon(module, config: dict) -> None:
    PatternRewriteWalker(GeluCanonPattern(config)).rewrite_module(module)
