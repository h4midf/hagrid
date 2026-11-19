"""quantize_per_tensor: one function per impl.

- "lut": a cinm.LUTOp on one 256-entry identity table (table[i] = i), shared by every
  op, indexed at x / scale + zero_point (alpha = 1 / scale, beta = zero_point).
- a circuit impl (quant_encoding.CIRCUIT_IMPLS, custom_op_library/quantize): the op
  stays, with its scale / zero point's (sh, m, zp) encoding stamped on it
  (quant_encoding.stamp_circuit_encoding).
"""
from xdsl.dialects.builtin import TensorType
from xdsl.pattern_rewriter import PatternRewriter, PatternRewriteWalker, RewritePattern, op_type_rewrite_pattern

from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance
from compiler.ops.dispatch import dispatch, get_entry
from compiler.ops.quant_encoding import CIRCUIT_IMPLS, QUANTIZE, stamp_circuit_encoding
from compiler.dialects.common_dialect import ui8

KIND = QUANTIZE


class _LutHandler:
    def __init__(self):
        self._table: ConstRefOp | None = None

    def __call__(self, op, entry: dict, rewriter: PatternRewriter) -> None:
        new_ops = []
        if self._table is None:
            self._table = ConstRefOp("quantize_identity_lut_n256", TensorType(ui8, [256]))
            new_ops.append(self._table)
        scale = float_from_prop(op.scale)
        zero_point = int_from_prop(op.zero_point)
        y = nd.CinmLUTOp(
            self._table.result, op.operand, op.result.type,
            alpha=1.0 / scale, beta=float(zero_point), op_tag="quantize_per_tensor",
        )
        new_ops.append(y)
        copy_provenance(op, y)
        rewriter.replace_matched_op(new_ops, new_results=y.results)


def _handle_circuit(op, entry: dict, rewriter: PatternRewriter) -> None:
    stamp_circuit_encoding(KIND, op, entry)


def build_impl_handlers() -> dict:
    return {"lut": _LutHandler(), **{impl: _handle_circuit for impl in CIRCUIT_IMPLS}}


class QuantizePerTensorCanonPattern(RewritePattern):
    def __init__(self, config: dict):
        self.config = config
        self.impl_handlers = build_impl_handlers()

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmQuantizePerTensorOp, rewriter: PatternRewriter) -> None:
        entry = get_entry(self.config, op, KIND)
        dispatch(op, entry, rewriter, self.impl_handlers)


def run_quantize_per_tensor_canon(module, config: dict) -> None:
    PatternRewriteWalker(QuantizePerTensorCanonPattern(config)).rewrite_module(module)
