"""relu: one function per impl.

- "lut": the shared activation LUT (_activation_lut_common.py).
- "exact": the op stays, lowered to cim.relu and costed on custom_op_library/relu (a
  comparator and mux).
"""
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.ops._activation_lut_common import FusedLutActivationHandler, no_suffix
from compiler.ops.dispatch import dispatch, get_entry

KIND = "relu"


def _handle_exact(op, entry: dict, rewriter: PatternRewriter) -> None:
    return


def build_impl_handlers() -> dict:
    return {"lut": FusedLutActivationHandler("relu", no_suffix), "exact": _handle_exact}


class ReluCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmReluOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_relu_canon(module, config: dict) -> None:
    PatternRewriteWalker(ReluCanonPattern(config)).rewrite_module(module)
