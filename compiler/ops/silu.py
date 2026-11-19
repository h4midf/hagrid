"""silu: one function per impl.

- "lut": the index x * alpha + beta as vec_el_mul / vec_add ops (not in the LUTOp's
  alpha / beta like the other activations), then a LUTOp; a new table per op.
- "exact": the op stays, lowered to cim.silu and costed on custom_op_library/silu
  (x * sigmoid(x), sigmoid = exp / (1 + exp)).
"""
from xdsl.dialects.builtin import TensorType, f32
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance
from compiler.ops.dispatch import dispatch, fx_name_of, get_entry, require

KIND = "silu"


def _handle_lut(op, entry: dict, rewriter: PatternRewriter) -> None:
    fx_name = fx_name_of(op)
    xmin = require(entry, "xmin", fx_name)
    xmax = require(entry, "xmax", fx_name)
    nbins = require(entry, "nbins", fx_name)
    attr_name = f"silu_lut_n{nbins}"
    table_op = ConstRefOp(attr_name, TensorType(f32, [nbins]))
    alpha = float((nbins - 1) / (xmax - xmin))
    beta = float(-xmin * alpha)

    operand_type = op.operand.type
    mul = nd.CinmVecElMulOp(op.operand, alpha, operand_type)
    copy_provenance(op, mul)
    idx = nd.CinmVecAddOp(mul.result, beta, operand_type)
    copy_provenance(op, idx)
    # the index map is applied by the mul/add above
    new_op = nd.CinmLUTOp(table_op.result, idx.result, op.result.type, alpha=1.0, beta=0.0)
    copy_provenance(op, new_op)
    rewriter.replace_matched_op([table_op, mul, idx, new_op], new_results=new_op.results)


def _handle_exact(op, entry: dict, rewriter: PatternRewriter) -> None:
    return


def build_impl_handlers() -> dict:
    return {"lut": _handle_lut, "exact": _handle_exact}


class SiluCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmSiluOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_silu_canon(module, config: dict) -> None:
    PatternRewriteWalker(SiluCanonPattern(config)).rewrite_module(module)
