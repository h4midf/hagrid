"""The cinm-stage canonicalization: run_full_cinm_canon runs the per-kind,
config-dispatched patterns (compiler/ops/<kind>.py) and the structural, config-free
patterns here (ReplaceLutPatternsPattern, DivAsVecMulPattern,
FoldReciprocalIntoLutPattern, MulMulScalarChainPattern, TransposeToPermutePattern,
ViewContiguousToReshapePattern) in order. run_quantize_lut_canon and
run_inline_shape_info_canon are separate entry points.
"""

from xdsl.dialects.builtin import ModuleOp, TensorType, f32
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import cinm_dialect as nd
from compiler.dialects import gen_dialect as gd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of
from compiler.ops import (
    dequantize, elu, gelu, group_norm, layer_norm, leaky_relu, quantize_per_tensor, relu, requantize,
    rms_norm, silu, softmax,
)

# cinm op class -> the op name ReplaceLutPatternsPattern builds its table names from
_CINM_OP_NAMES = {
    nd.CinmVecElMulOp: "vec_el_mul",
    nd.CinmLUTOp: "LUTOp",
    nd.CinmLUTOpFused: "LUTOp",
    nd.CinmReduceVecOp: "reduceVec",
    nd.CinmNegOp: "neg",
    nd.CinmCatOp: "cat",
    nd.CinmDotProductOp: "dotProductOp",
    nd.CinmVecAddOp: "vec_add",
    nd.CinmScalarAddOp: "scalar_add",
    nd.CinmTransposeOp: "transpose",
    nd.CinmPermuteOp: "permute",
    nd.CinmViewOp: "view",
    nd.CinmReshapeOp: "reshape",
    nd.CinmContiguousOp: "contiguous",
    nd.CinmBroadcastOp: "broadcast",
    nd.CinmUnfoldOp: "unfold",
    nd.CinmMaxOp: "max",
    nd.CinmSqrtOp: "sqrt",
    nd.CinmReluOp: "relu",
    nd.CinmGeluOp: "gelu",
    nd.CinmSiluOp: "silu",
    nd.CinmEluOp: "elu",
    nd.CinmLeakyReluOp: "leaky_relu",
    nd.CinmSoftmaxOp: "softmax",
    nd.CinmScalarDivOp: "scalar_div",
    nd.CinmVecDivOp: "vec_div",
    nd.CinmTruedivOp: "truediv",
    nd.CinmReciprocalOp: "reciprocal",
}


def _fx_op_name(op) -> str:
    """The op name of a cinm op for table names (e.g. CinmScalarAddOp -> "scalar_add")."""
    return _CINM_OP_NAMES[type(op)]


def _fnum(value: float) -> str:
    """A float in buffer-name form (-6.0 -> "-6p0")."""
    return str(value).replace(".", "p")


class MulMulScalarChainPattern(RewritePattern):
    """vec_el_mul(vec_el_mul(x, s1), s2) -> vec_el_mul(x, s1*s2), when both are a
    tensor times a literal scalar. Folds one level per match (it runs once, so a
    3-deep chain is not fully collapsed)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmVecElMulOp, rewriter: PatternRewriter) -> None:
        outer = _tensor_scalar_pair(op)
        if outer is None:
            return
        inner_value, s2_value = outer
        inner = inner_value.owner
        if not isinstance(inner, nd.CinmVecElMulOp):
            return
        inner_pair = _tensor_scalar_pair(inner)
        if inner_pair is None:
            return
        x, s1_value = inner_pair
        fused = nd.CinmVecElMulOp(x, float(s1_value * s2_value), op.result.type)
        copy_provenance(op, fused)
        rewriter.replace_matched_op(fused)
        # replace_matched_op erases only the outer op; the inner one is now dead
        if not inner.results[0].uses:
            rewriter.erase_op(inner)


def _tensor_scalar_pair(op: nd.CinmVecElMulOp):
    """(tensor, scalar) of a tensor-times-literal vec_el_mul, else None."""
    if op.rhs_scalar is not None:
        return op.lhs, float_from_prop(op.rhs_scalar)
    return None


class DivAsVecMulPattern(RewritePattern):
    """scalar_div / vec_div(x, y) -> vec_el_mul(x, reciprocal(y)), or
    vec_el_mul(x, 1/y) for a literal y. A literal x raises (LayerNorm's 1/sqrt(...)
    is consumed by the rsqrt-chain pattern first)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmScalarDivOp | nd.CinmVecDivOp, rewriter: PatternRewriter) -> None:
        if op.lhs is None:
            raise NotImplementedError("DivAsVecMulPattern: division with a scalar lhs is not supported")
        x = op.lhs
        new_ops = []
        if op.rhs is not None:
            recip = nd.CinmReciprocalOp(op.rhs, op.rhs.type)
            new_ops.append(recip)
            rhs_value = recip.result
        else:
            rhs_value = 1.0 / float_from_prop(op.rhs_scalar)
        mul = nd.CinmVecElMulOp(x, rhs_value, op.result.type)
        new_ops.append(mul)
        copy_provenance(op, mul)
        rewriter.replace_matched_op(new_ops, new_results=mul.results)


class FoldReciprocalIntoLutPattern(RewritePattern):
    """reciprocal(LUTOp(table, ...)) -> LUTOp(table_recip, ...): the same lookup into
    a "{table}_recip" table (the contents change, not the index mapping). Raises if
    the LUT's table is not a const.ref."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmReciprocalOp, rewriter: PatternRewriter) -> None:
        src = op.operand.owner
        if not isinstance(src, (nd.CinmLUTOp, nd.CinmLUTOpFused)):
            return
        table_op = src.table.owner
        if not isinstance(table_op, ConstRefOp):
            raise RuntimeError(
                "FoldReciprocalIntoLutPattern: LUT config mismatch; expected const.ref at LUTOp's table operand"
            )
        recip_table = ConstRefOp(f"{table_op.sym_name.data}_recip", table_op.result.type)
        if isinstance(src, nd.CinmLUTOp):
            new_lut = nd.CinmLUTOp(
                recip_table.result, src.index, op.result.type,
                alpha=float_from_prop(src.alpha), beta=float_from_prop(src.beta),
            )
        else:
            new_lut = nd.CinmLUTOpFused(recip_table.result, list(src.args), op.result.type)
        copy_provenance(op, new_lut)
        rewriter.replace_matched_op([recip_table, new_lut], new_results=new_lut.results)
        if not src.results[0].uses:
            rewriter.erase_op(src)
            if not table_op.results[0].uses:
                rewriter.erase_op(table_op)


def _lut_operand_type(op) -> str:
    """"TENSOR" if the op's first operand has a last dim != 1, else "SCALAR"
    (also with no operand). Independent of the op's scalar_* / vec_* name."""
    for operand in getattr(op, "operands", ()):
        if operand is None:
            continue
        shape = shape_of(operand.type)
        return "TENSOR" if shape and shape[-1] != 1 else "SCALAR"
    return "SCALAR"


def _match_sqrt_div_add_chain(op):
    """[scalar_add, scalar_div, sqrt] if op is scalar_add(scalar_div(sqrt(...))), all
    SCALAR, else None. (Not LayerNorm's rsqrt chain, where sqrt consumes the add.)"""
    if not isinstance(op, nd.CinmScalarAddOp) or _lut_operand_type(op) != "SCALAR":
        return None
    for div_candidate in (op.lhs, op.rhs):
        if div_candidate is None:
            continue
        div_op = div_candidate.owner
        if not isinstance(div_op, nd.CinmScalarDivOp) or _lut_operand_type(div_op) != "SCALAR":
            continue
        for sqrt_candidate in (div_op.lhs, div_op.rhs):
            if sqrt_candidate is None:
                continue
            sqrt_op = sqrt_candidate.owner
            if isinstance(sqrt_op, nd.CinmSqrtOp) and _lut_operand_type(sqrt_op) == "SCALAR":
                return [op, div_op, sqrt_op]
    return None


def _match_lut_div_chain(op, div_cls, div_operand_type):
    """[div, lut] if op is a div_cls (of div_operand_type) of a TENSOR LUT result,
    else None."""
    if not isinstance(op, div_cls) or _lut_operand_type(op) != div_operand_type:
        return None
    for operand in (op.lhs, op.rhs):
        if operand is not None and isinstance(operand.owner, (nd.CinmLUTOp, nd.CinmLUTOpFused)):
            lut_op = operand.owner
            if _lut_operand_type(lut_op) == "TENSOR":
                return [op, lut_op]
    return None


class ReplaceLutPatternsPattern(RewritePattern):
    """Fuses one of three disjoint chains (_match_*_chain) into one N-ary
    LUTOp(lut_config_<ops>, *leftover operands). The fused table is a name and a
    type only: its contents are not defined by the matched chain."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: nd.CinmScalarAddOp | nd.CinmScalarDivOp | nd.CinmVecDivOp, rewriter: PatternRewriter
    ) -> None:
        matched = _match_sqrt_div_add_chain(op)
        if matched is None:
            matched = _match_lut_div_chain(op, nd.CinmScalarDivOp, "SCALAR")
        if matched is None:
            matched = _match_lut_div_chain(op, nd.CinmVecDivOp, "TENSOR")
        if matched is None:
            return

        terminal = matched[-1]
        if isinstance(terminal, (nd.CinmLUTOp, nd.CinmLUTOpFused)) and len(terminal.operands) >= 2:
            leftover = [terminal.args[0] if isinstance(terminal, nd.CinmLUTOpFused) else terminal.index]
        else:
            matched_set = set(matched)
            leftover = []
            for n in matched:
                for operand in getattr(n, "operands", ()):
                    if operand is not None and operand.owner not in matched_set:
                        leftover.append(operand)

        # a fixed table size: this structural fusion has no config entry of its own
        nbins = 256
        attr_name = "lut_config_" + "_".join(_fx_op_name(n) for n in matched)
        table = ConstRefOp(attr_name, TensorType(f32, [nbins]))
        new_op = nd.CinmLUTOpFused(table.result, leftover, op.result.type)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op([table, new_op], new_results=new_op.results)
        for old in matched[1:]:
            if not old.results[0].uses:
                rewriter.erase_op(old)


class TransposeToPermutePattern(RewritePattern):
    """transpose(x, 1, 2) of a rank-4 tensor -> permute(x, 0, 2, 1, 3). Skips weight
    (const.ref) transposes and other ranks."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmTransposeOp, rewriter: PatternRewriter) -> None:
        if int_from_prop(op.dim0) != 1 or int_from_prop(op.dim1) != 2:
            return
        src_owner = op.operand.owner
        if isinstance(src_owner, ConstRefOp):
            return
        if len(shape_of(op.operand.type)) != 4:
            return
        new_op = nd.CinmPermuteOp(op.operand, [0, 2, 1, 3], op.result.type)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


class ViewContiguousToReshapePattern(RewritePattern):
    """view(contiguous(x), shape) -> reshape(x, shape)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmViewOp, rewriter: PatternRewriter) -> None:
        c = op.operand.owner
        if not isinstance(c, nd.CinmContiguousOp):
            return
        new_op = nd.CinmReshapeOp(c.operand, shape_of(op.result.type), op.result.type)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


def run_quantize_lut_canon(module: ModuleOp, config: dict) -> None:
    """In place: cinm.quantize_per_tensor / cinm.dequantize, config-dispatched by
    compiler/ops/quantize_per_tensor.py and dequantize.py, and the requantize of each
    cinm.quantized_matmul (compiler/ops/requantize.py). Disjoint from
    run_full_cinm_canon's ops, so the order between them does not matter."""
    PatternRewriteWalker(
        GreedyRewritePatternApplier([
            quantize_per_tensor.QuantizePerTensorCanonPattern(config),
            dequantize.DequantizeCanonPattern(config),
            requantize.RequantizeCanonPattern(config),
        ])
    ).rewrite_module(module)


def run_reciprocal_chain_canon(module: ModuleOp, config: dict) -> None:
    """In place, as four separate non-recursive passes in this order:
    RsqrtChainCanonPattern, DivAsVecMulPattern, GeneralReciprocalCanonPattern,
    FoldReciprocalIntoLutPattern. Each pass does not revisit ops it inserts, but a
    later pass sees an earlier one's (e.g. DivAsVecMulPattern's new reciprocals).
    The config-dispatched two are in compiler/ops/layer_norm.py."""
    PatternRewriteWalker(layer_norm.RsqrtChainCanonPattern(config), apply_recursively=False).rewrite_module(module)
    PatternRewriteWalker(DivAsVecMulPattern(), apply_recursively=False).rewrite_module(module)
    PatternRewriteWalker(layer_norm.GeneralReciprocalCanonPattern(config), apply_recursively=False).rewrite_module(module)
    PatternRewriteWalker(FoldReciprocalIntoLutPattern(), apply_recursively=False).rewrite_module(module)


def run_replace_lut_patterns_canon(module: ModuleOp) -> None:
    """ReplaceLutPatternsPattern to a fixpoint (apply_recursively, the default),
    until no chain matches."""
    PatternRewriteWalker(ReplaceLutPatternsPattern()).rewrite_module(module)


def run_cinm_canon(module: ModuleOp) -> None:
    """In place: MulMulScalarChainPattern, then TransposeToPermutePattern and
    ViewContiguousToReshapePattern."""
    PatternRewriteWalker(MulMulScalarChainPattern(), apply_recursively=False).rewrite_module(module)
    PatternRewriteWalker(
        GreedyRewritePatternApplier([TransposeToPermutePattern(), ViewContiguousToReshapePattern()])
    ).rewrite_module(module)


def run_lut_activation_canon(module: ModuleOp, config: dict) -> None:
    """In place: gelu, silu, relu, elu, leaky_relu (disjoint op types, so their order
    does not matter). Softmax runs as its own pass after these."""
    PatternRewriteWalker(
        GreedyRewritePatternApplier([
            gelu.GeluCanonPattern(config),
            silu.SiluCanonPattern(config),
            relu.ReluCanonPattern(config),
            elu.EluCanonPattern(config),
            leaky_relu.LeakyReluCanonPattern(config),
        ])
    ).rewrite_module(module)


def run_softmax_canon(module: ModuleOp, config: dict) -> None:
    PatternRewriteWalker(softmax.SoftmaxCanonPattern(config)).rewrite_module(module)


def run_full_cinm_canon(module: ModuleOp, config: dict) -> None:
    """In place, in order: the activations, softmax, the LUT-chain fusion fixpoint,
    the norm decompositions, the reciprocal-chain passes, then the mul-chain and
    layout passes.

    config (per fx_name) is required: every op of a config-dispatched kind must have
    an entry, or its compiler/ops/<kind>.py pattern raises."""
    run_lut_activation_canon(module, config)
    run_softmax_canon(module, config)
    run_replace_lut_patterns_canon(module)
    # the norm decompositions must run before run_reciprocal_chain_canon: their "lut"
    # impl builds the sqrt / reciprocal ops it rewrites (rms_norm's a bare
    # reciprocal, the others an rsqrt chain)
    layer_norm.run_layer_norm_decompose_canon(module, config)
    group_norm.run_group_norm_decompose_canon(module, config)
    rms_norm.run_rms_norm_decompose_canon(module, config)
    run_reciprocal_chain_canon(module, config)
    run_cinm_canon(module)
    PatternRewriteWalker(
        GreedyRewritePatternApplier([TransposeToPermutePattern(), ViewContiguousToReshapePattern()])
    ).rewrite_module(module)


class ReshapeAsToReshapePattern(RewritePattern):
    """gen.reshape_as(a, b) -> gen.reshape(a, shape(b)), dropping the edge on b."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenReshapeAsOp, rewriter: PatternRewriter) -> None:
        y = gd.GenReshapeOp(op.operand, shape_of(op.like.type), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


def run_inline_shape_info_canon(module: ModuleOp) -> None:
    PatternRewriteWalker(ReshapeAsToReshapePattern()).rewrite_module(module)
