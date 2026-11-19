"""elu: one function per impl.

- "lut": the shared activation LUT (_activation_lut_common.py), one table per alpha.
- "exact": the op stays, lowered to cim.elu and costed on custom_op_library/elu (one
  circuit per alpha).
"""
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop
from compiler.ops._activation_lut_common import FusedLutActivationHandler, fnum
from compiler.ops.dispatch import dispatch, get_entry

KIND = "elu"


def _elu_suffix(op) -> str:
    return f"_a{fnum(float_from_prop(op.alpha))}"


def _handle_exact(op, entry: dict, rewriter: PatternRewriter) -> None:
    return


def build_impl_handlers() -> dict:
    return {"lut": FusedLutActivationHandler("elu", _elu_suffix), "exact": _handle_exact}


class EluCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmEluOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_elu_canon(module, config: dict) -> None:
    PatternRewriteWalker(EluCanonPattern(config)).rewrite_module(module)
