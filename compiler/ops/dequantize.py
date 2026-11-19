"""dequantize: one function per impl.

- "lut": a cinm.LUTOp on a 256-entry table, table[i] = (i - zero_point) * scale, indexed
  by the int8 operand itself. Its values depend on the op's scale and zero point, so
  each op gets its own table.
- a circuit impl (quant_encoding.CIRCUIT_IMPLS, custom_op_library/dequantize): the op
  stays, with its scale / zero point's (sh, m, zp) encoding stamped on it
  (quant_encoding.stamp_circuit_encoding).
"""
from xdsl.dialects.builtin import TensorType, f32
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance
from compiler.ops.dispatch import dispatch, get_entry
from compiler.ops.quant_encoding import CIRCUIT_IMPLS, DEQUANTIZE, stamp_circuit_encoding

KIND = DEQUANTIZE


class _LutHandler:
    def __init__(self):
        self._counter = 0

    def __call__(self, op, entry: dict, rewriter: PatternRewriter) -> None:
        table = ConstRefOp(f"dequantize_lut_n256_{self._counter}", TensorType(f32, [256]))
        self._counter += 1
        # the operand is the table index (0-255): identity index map
        y = nd.CinmLUTOp(table.result, op.operand, op.result.type, alpha=1.0, beta=0.0, op_tag="dequantize")
        copy_provenance(op, y)
        rewriter.replace_matched_op([table, y], new_results=y.results)


def _handle_circuit(op, entry: dict, rewriter: PatternRewriter) -> None:
    stamp_circuit_encoding(KIND, op, entry)


def build_impl_handlers() -> dict:
    return {"lut": _LutHandler(), **{impl: _handle_circuit for impl in CIRCUIT_IMPLS}}


class DequantizeCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmDequantizeOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_dequantize_canon(module, config: dict) -> None:
    PatternRewriteWalker(DequantizeCanonPattern(config)).rewrite_module(module)
