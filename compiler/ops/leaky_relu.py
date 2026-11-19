"""leaky_relu: one function per impl.

- "lut": the shared activation LUT (_activation_lut_common.py), one table per
  negative_slope.
- "exact": the op stays, lowered to cim.leaky_relu and costed on
  custom_op_library/leaky_relu (one circuit per negative_slope).
"""
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop
from compiler.ops._activation_lut_common import FusedLutActivationHandler, fnum
from compiler.ops.dispatch import dispatch, get_entry

KIND = "leaky_relu"


def _leaky_relu_suffix(op) -> str:
    return f"_s{fnum(float_from_prop(op.negative_slope))}"


def _handle_exact(op, entry: dict, rewriter: PatternRewriter) -> None:
    return


def build_impl_handlers() -> dict:
    return {"lut": FusedLutActivationHandler("leaky_relu", _leaky_relu_suffix), "exact": _handle_exact}


class LeakyReluCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmLeakyReluOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_leaky_relu_canon(module, config: dict) -> None:
    PatternRewriteWalker(LeakyReluCanonPattern(config)).rewrite_module(module)
