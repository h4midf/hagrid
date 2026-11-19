"""The gen -> cinm lowering, in two steps (both run by compiler/lowering.py):

run_rewrite_cinm_applicable_canon: the module-level ops that need more than a 1:1
  rewrite (LayerNorm, GroupNorm/InstanceNorm, RMSNorm, norm, Embedding, ReLU,
  LeakyReLU, MaxPool2d, AdaptiveAvgPool2d, Upsample).
run_gen_to_cinm_canon: every remaining gen op to its cinm op (quantize/dequantize,
  quantized Linear/matmul, softmax, gelu, silu, elu, sqrt, add/mul/div, layout ops,
  cat, slice, chunk, mean, argmax/max, scatter).

The norms and activations stay one atomic cinm op here; compiler/ops/<kind>.py later
decomposes them or keeps them for a circuit, per the variant config. A gen.norm with
p != 2 is left unmatched. Every matmul is int8 (the bridge rejects float
matmul / Linear / conv); a quantized Linear's matmul gets logical_matmul_* attributes
naming its module.
"""

from xdsl.dialects.builtin import IntegerType, Signedness, StringAttr, TensorType, f32
from xdsl.ir import SSAValue
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import cinm_dialect as nd
from compiler.dialects import gen_dialect as gd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop, int_list_from_prop
from compiler.dialects.common_dialect import SLICE_NONE, CommonSliceOp, ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, require_lowered, scalar_rhs_of, shape_of


class LayerNormAsCinmPattern(RewritePattern):
    """gen.layer_norm -> one atomic cinm.layer_norm. compiler/ops/layer_norm.py
    later decomposes it ("lut") or keeps it for the circuit, per the config."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenLayerNormOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmLayerNormOp(
            op.operand, op.result.type, hidden_size=int_from_prop(op.hidden_size), eps=float_from_prop(op.eps),
            weight_name=op.weight_name.data if op.weight_name is not None else None,
            bias_name=op.bias_name.data if op.bias_name is not None else None,
        )
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ReLUAsCinmPattern(RewritePattern):
    """gen.relu -> cinm.relu (compiler/ops/relu.py later picks LUT or circuit)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenReLUOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmReluOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class EmbeddingAsCinmPattern(RewritePattern):
    """gen.embedding -> const.ref(table) [num_embeddings, embedding_dim] +
    cinm.embedding (its own op, not cinm.LUTOp; see CinmEmbeddingOp)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenEmbeddingOp, rewriter: PatternRewriter) -> None:
        elem_ty = op.result.type.element_type
        embedding_dim = shape_of(op.result.type)[-1]
        num_embeddings = int_from_prop(op.num_embeddings)
        table = ConstRefOp(op.weight_name.data, TensorType(elem_ty, [num_embeddings, embedding_dim]))
        y = nd.CinmEmbeddingOp(table.result, op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op([table, y], new_results=y.results)


class MaxPool2dAsMaxPattern(RewritePattern):
    """cinm.unfold -> reshape to split channels out of C*kH*kW (pooling does not mix
    channels) -> transpose the window axis last -> cinm.max (keepdim=False) ->
    reshape to the output shape."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenMaxPool2dOp, rewriter: PatternRewriter) -> None:
        x = op.operand
        elem_ty = x.type.element_type
        out_shape = shape_of(op.result.type)
        n, channels, h_out, w_out = out_shape
        l = h_out * w_out
        kernel_size = int_list_from_prop(op.kernel_size)
        kh, kw = kernel_size

        unfolded = nd.CinmUnfoldOp(
            x, kernel_size, TensorType(elem_ty, [n, channels * kh * kw, l]),
            dilation=int_list_from_prop(op.dilation), padding=int_list_from_prop(op.padding), stride=int_list_from_prop(op.stride),
        )
        split = nd.CinmReshapeOp(unfolded.result, [n, channels, kh * kw, l], TensorType(elem_ty, [n, channels, kh * kw, l]))
        windows_last = gd.GenTransposeOp(split.result, -2, -1, TensorType(elem_ty, [n, channels, l, kh * kw]))
        pooled = nd.CinmMaxOp(windows_last.result, TensorType(elem_ty, [n, channels, l]), reduce_dim=-1, k=1)
        copy_provenance(op, pooled)
        y = nd.CinmReshapeOp(pooled.result, out_shape, op.result.type)

        copy_provenance(op, y)
        rewriter.replace_matched_op([unfolded, split, windows_last, pooled, y], new_results=y.results)


class AdaptiveAvgPool2dAsMeanPattern(RewritePattern):
    """reshape -> reduce_vec (sum) -> vec_el_mul (1/(H*W)) -> reshape.
    output_size=(1, 1) only."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenAdaptiveAvgPool2dOp, rewriter: PatternRewriter) -> None:
        output_size = tuple(int_list_from_prop(op.output_size))
        if output_size != (1, 1):
            raise NotImplementedError("AdaptiveAvgPool2dAsMeanPattern: only output_size=(1, 1) is supported")

        x = op.operand
        elem_ty = x.type.element_type
        n, c, h, w = shape_of(x.type)
        inv_hw = 1.0 / float(h * w)

        flattened = nd.CinmReshapeOp(x, [n, c, h * w], TensorType(elem_ty, [n, c, h * w]))
        summed = nd.CinmReduceVecOp(flattened.result, TensorType(elem_ty, [n, c, 1]))
        copy_provenance(op, summed)
        scaled = nd.CinmVecElMulOp(summed.result, inv_hw, TensorType(elem_ty, [n, c, 1]))
        copy_provenance(op, scaled)
        y = nd.CinmReshapeOp(scaled.result, [n, c, 1, 1], op.result.type)

        copy_provenance(op, y)
        rewriter.replace_matched_op([flattened, summed, scaled, y], new_results=y.results)


class MeanAsCinmPattern(RewritePattern):
    """gen.mean(dim) -> permute dim to the last axis (skipped if it already is) ->
    cinm.reduce_vec (sum, last axis only) -> cinm.vec_el_mul (1/N) -> reshape (drop
    the size-1 axis). Moving only dim to the end keeps the other axes in order, so
    no permute back is needed."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenMeanOp, rewriter: PatternRewriter) -> None:
        in_shape = shape_of(op.operand.type)
        rank = len(in_shape)
        dim = int_from_prop(op.dim)
        if dim < 0:
            dim += rank
        elem_ty = op.operand.type.element_type

        new_ops = []
        cur = op.operand
        cur_shape = list(in_shape)
        if dim != rank - 1:
            perm = [i for i in range(rank) if i != dim] + [dim]
            cur_shape = [in_shape[i] for i in perm]
            permuted = nd.CinmPermuteOp(cur, perm, TensorType(elem_ty, cur_shape))
            new_ops.append(permuted)
            cur = permuted.result

        reduce_size = cur_shape[-1]
        reduced_shape = cur_shape[:-1] + [1]
        summed = nd.CinmReduceVecOp(cur, TensorType(elem_ty, reduced_shape))
        new_ops.append(summed)
        copy_provenance(op, summed)
        scaled = nd.CinmVecElMulOp(summed.result, 1.0 / float(reduce_size), TensorType(elem_ty, reduced_shape))
        new_ops.append(scaled)
        copy_provenance(op, scaled)
        squeezed = nd.CinmReshapeOp(scaled.result, cur_shape[:-1], op.result.type)
        new_ops.append(squeezed)

        copy_provenance(op, squeezed)
        rewriter.replace_matched_op(new_ops, new_results=squeezed.results)


class ArgmaxAsCinmPattern(RewritePattern):
    """gen.argmax -> cinm.argmax (operand, dim, k; k > 1 is top-k)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenArgmaxOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmArgmaxOp(op.operand, op.result.type, reduce_dim=int_from_prop(op.dim), k=int_from_prop(op.k))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class MaxAsCinmPattern(RewritePattern):
    """gen.max -> cinm.max: the values counterpart of ArgmaxAsCinmPattern."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenMaxOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmMaxOp(op.operand, op.result.type, reduce_dim=int_from_prop(op.dim), k=int_from_prop(op.k))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ScatterAsCinmPattern(RewritePattern):
    """gen.scatter -> cinm.scatter, only when the base is a gen.zeros_like (an
    all-zero base needs no operand); any other base is left unmatched. Erases the
    zeros_like once unused."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenScatterOp, rewriter: PatternRewriter) -> None:
        base_op = op.base.owner
        if not isinstance(base_op, gd.GenZerosLikeOp):
            return
        y = nd.CinmScatterOp(op.index, op.src, op.result.type, dim=int_from_prop(op.dim))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)
        if not base_op.results[0].uses:
            rewriter.erase_op(base_op)


def _match_rms_norm_shape(mul_op: gd.GenMulOp):
    """A hand-written RMSNorm module is traced inline as
    `weight * (x / ((x.norm(2, dim=-1, keepdim=True) / sqrt(DIM)) + eps))`. Walks
    back from a GenMulOp through that shape; returns (x, weight_const, eps) or None.
    A gen.norm outside this shape is left for NormAsCinmPattern."""
    if mul_op.rhs is None:
        return None  # scalar-rhs form: weight*(tensor) needs a real rhs operand
    lhs_owner, rhs_owner = mul_op.lhs.owner, mul_op.rhs.owner
    if isinstance(lhs_owner, ConstRefOp) and not isinstance(rhs_owner, ConstRefOp):
        weight_const, div2 = lhs_owner, rhs_owner
    elif isinstance(rhs_owner, ConstRefOp) and not isinstance(lhs_owner, ConstRefOp):
        weight_const, div2 = rhs_owner, lhs_owner
    else:
        return None

    if not isinstance(div2, gd.GenDivOp):
        return None
    if div2.lhs is None or div2.rhs is None or div2.lhs_scalar is not None or div2.rhs_scalar is not None:
        return None  # x / (rms+eps): both real tensor operands, no scalar
    x = div2.lhs
    add_op = div2.rhs.owner
    if not isinstance(add_op, gd.GenAddOp):
        return None
    if add_op.lhs is None or add_op.rhs is not None or add_op.lhs_scalar is not None or add_op.rhs_scalar is None:
        return None  # rms + eps: tensor lhs, scalar rhs
    eps = float_from_prop(add_op.rhs_scalar)
    div1 = add_op.lhs.owner
    if not isinstance(div1, gd.GenDivOp):
        return None
    if div1.lhs is None or div1.rhs is not None or div1.lhs_scalar is not None or div1.rhs_scalar is None:
        return None  # norm / sqrt(DIM): tensor lhs, scalar rhs
    norm_op = div1.lhs.owner
    if not isinstance(norm_op, gd.GenNormOp):
        return None
    if int_from_prop(norm_op.p) != 2 or norm_op.operand is not x:
        return None
    return x, weight_const, eps


class RMSNormAsCinmPattern(RewritePattern):
    """The 5-op RMSNorm shape (_match_rms_norm_shape) -> one atomic cinm.rms_norm,
    like LayerNormAsCinmPattern."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenMulOp, rewriter: PatternRewriter) -> None:
        shape = _match_rms_norm_shape(op)
        if shape is None:
            return
        x, weight_const, eps = shape
        hidden = shape_of(x.type)[-1]
        y = nd.CinmRMSNormOp(
            x, op.result.type, hidden_size=hidden, eps=eps,
            weight_name=weight_const.sym_name.data if weight_const is not None else None,
        )
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


def run_rms_norm_canon(module) -> None:
    PatternRewriteWalker(RMSNormAsCinmPattern()).rewrite_module(module)


class NormAsCinmPattern(RewritePattern):
    """gen.norm with p=2 -> cinm.dot_product(x, x) -> gen.sqrt (run_gen_to_cinm_canon
    lowers it). Any other p is left unmatched."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenNormOp, rewriter: PatternRewriter) -> None:
        if int_from_prop(op.p) != 2:
            return
        dot = nd.CinmDotProductOp(op.operand, op.operand, op.result.type)
        y = gd.GenSqrtOp(dot.result, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op([dot, y], new_results=y.results)


def run_norm_canon(module) -> None:
    PatternRewriteWalker(NormAsCinmPattern()).rewrite_module(module)


def run_layer_norm_canon(module) -> None:
    PatternRewriteWalker(LayerNormAsCinmPattern()).rewrite_module(module)


def run_relu_canon(module) -> None:
    PatternRewriteWalker(ReLUAsCinmPattern()).rewrite_module(module)


def run_embedding_canon(module) -> None:
    PatternRewriteWalker(EmbeddingAsCinmPattern()).rewrite_module(module)


def run_max_pool2d_canon(module) -> None:
    PatternRewriteWalker(MaxPool2dAsMaxPattern()).rewrite_module(module)


def run_adaptive_avg_pool2d_canon(module) -> None:
    PatternRewriteWalker(AdaptiveAvgPool2dAsMeanPattern()).rewrite_module(module)


class GroupLikeNormAsCinmPattern(RewritePattern):
    """gen.group_like_norm -> one atomic cinm.group_like_norm, like
    LayerNormAsCinmPattern. compiler/ops/group_norm.py decomposes it ("lut" is its
    only impl; no circuit exists)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenGroupLikeNormOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmGroupLikeNormOp(
            op.operand, op.result.type, num_groups=int_from_prop(op.num_groups), channels=int_from_prop(op.channels),
            eps=float_from_prop(op.eps),
            weight_name=op.weight_name.data if op.weight_name is not None else None,
            bias_name=op.bias_name.data if op.bias_name is not None else None,
        )
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class LeakyReLUAsCinmPattern(RewritePattern):
    """gen.leaky_relu -> cinm.leaky_relu."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenLeakyReluOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmLeakyReluOp(op.operand, op.result.type, negative_slope=float_from_prop(op.negative_slope))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


def run_group_like_norm_canon(module) -> None:
    PatternRewriteWalker(GroupLikeNormAsCinmPattern()).rewrite_module(module)


def run_leaky_relu_canon(module) -> None:
    PatternRewriteWalker(LeakyReLUAsCinmPattern()).rewrite_module(module)


class UpsampleAsCinmPattern(RewritePattern):
    """reshape -> broadcast -> reshape (nearest-neighbor duplication). The bridge
    allows only mode='nearest' with an integer scale_factor."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenUpsampleOp, rewriter: PatternRewriter) -> None:
        x = op.operand
        elem_ty = x.type.element_type
        n, c, h, w = shape_of(x.type)
        sh = int_from_prop(op.scale_h)
        sw = int_from_prop(op.scale_w)

        split = nd.CinmReshapeOp(x, [n, c, h, 1, w, 1], TensorType(elem_ty, [n, c, h, 1, w, 1]))
        expanded = nd.CinmBroadcastOp(split.result, [n, c, h, sh, w, sw], TensorType(elem_ty, [n, c, h, sh, w, sw]))
        y = nd.CinmReshapeOp(expanded.result, [n, c, h * sh, w * sw], op.result.type)

        copy_provenance(op, y)
        rewriter.replace_matched_op([split, expanded, y], new_results=y.results)


def run_upsample_canon(module) -> None:
    PatternRewriteWalker(UpsampleAsCinmPattern()).rewrite_module(module)


def _numel(shape: list) -> int:
    n = 1
    for d in shape:
        n *= d
    return n


def _scalar_rhs_op_is_scalar_like(op) -> bool:
    """For a binary_op_scalar_rhs op: scalar-like if the rhs is a literal, or either
    side has exactly one element."""
    if op.rhs_scalar is not None:
        return True
    if op.rhs is not None and _numel(shape_of(op.rhs.type)) == 1:
        return True
    return _numel(shape_of(op.lhs.type)) == 1


def _either_scalar_op_is_scalar_like(op) -> bool:
    """_scalar_rhs_op_is_scalar_like for a binary_op_either_scalar op, where either
    side can be a literal."""
    if op.lhs_scalar is not None or op.rhs_scalar is not None:
        return True
    if op.lhs is not None and _numel(shape_of(op.lhs.type)) == 1:
        return True
    if op.rhs is not None and _numel(shape_of(op.rhs.type)) == 1:
        return True
    return False


class SoftmaxAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenSoftmaxOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmSoftmaxOp(op.operand, op.result.type, dim=int_from_prop(op.dim))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


def _copy_gelu_form(src_op, dst_op) -> None:
    """The GELU form ("approximate": "none" / "tanh", set by the frontend) travels with
    the op: it picks the circuit (compiler/ops/gelu.py)."""
    dst_op.attributes["approximate"] = src_op.attributes["approximate"]


class GeluAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenGeluOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmGeluOp(op.operand, op.result.type)
        copy_provenance(op, y)
        _copy_gelu_form(op, y)
        rewriter.replace_matched_op(y)


class SiluAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenSiluOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmSiluOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class SiluModuleAsCinmPattern(RewritePattern):
    """gen.silu_module (nn.SiLU) -> the same cinm.silu as F.silu."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenSiluModuleOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmSiluOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class GeluModuleAsCinmPattern(RewritePattern):
    """gen.gelu_module (nn.GELU) -> the same cinm.gelu as F.gelu."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenGeluModuleOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmGeluOp(op.operand, op.result.type)
        copy_provenance(op, y)
        _copy_gelu_form(op, y)
        rewriter.replace_matched_op(y)


class EluAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenEluOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmEluOp(op.operand, op.result.type, alpha=float_from_prop(op.alpha))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class SqrtAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenSqrtOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmSqrtOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class TransposeAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenTransposeOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmTransposeOp(op.operand, int_from_prop(op.dim0), int_from_prop(op.dim1), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class AddAsCinmPattern(RewritePattern):
    """gen.add, and also cinm.vec_add / cinm.scalar_add built by earlier passes, ->
    cinm.scalar_add if scalar-like (a literal or a one-element side), else
    cinm.vec_add: the scalar-vs-vector choice is re-derived for every add. gen.add can
    have its scalar on the left; it is moved to the right (addition commutes)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: gd.GenAddOp | nd.CinmVecAddOp | nd.CinmScalarAddOp, rewriter: PatternRewriter
    ) -> None:
        if isinstance(op, gd.GenAddOp):
            lhs = op.lhs if op.lhs is not None else float_from_prop(op.lhs_scalar)
            rhs = op.rhs if op.rhs is not None else float_from_prop(op.rhs_scalar)
            is_scalar = _either_scalar_op_is_scalar_like(op)
            if not isinstance(lhs, SSAValue):
                lhs, rhs = rhs, lhs
        else:
            lhs = op.lhs
            rhs = scalar_rhs_of(op)
            is_scalar = _scalar_rhs_op_is_scalar_like(op)
        cls = nd.CinmScalarAddOp if is_scalar else nd.CinmVecAddOp
        y = cls(lhs, rhs, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class MulAsCinmPattern(RewritePattern):
    """gen.mul -> cinm.vec_el_mul, with no scalar-vs-vector split (unlike add/div)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenMulOp, rewriter: PatternRewriter) -> None:
        rhs = scalar_rhs_of(op)
        y = nd.CinmVecElMulOp(op.lhs, rhs, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class DivAsCinmPattern(RewritePattern):
    """gen.div, cinm.vec_div and cinm.scalar_div -> cinm.scalar_div or cinm.vec_div,
    re-derived like AddAsCinmPattern."""

    @op_type_rewrite_pattern
    def match_and_rewrite(
        self, op: gd.GenDivOp | nd.CinmVecDivOp | nd.CinmScalarDivOp, rewriter: PatternRewriter
    ) -> None:
        lhs = op.lhs if op.lhs is not None else float_from_prop(op.lhs_scalar)
        rhs = op.rhs if op.rhs is not None else float_from_prop(op.rhs_scalar)
        cls = nd.CinmScalarDivOp if _either_scalar_op_is_scalar_like(op) else nd.CinmVecDivOp
        y = cls(lhs, rhs, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class PermuteAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenPermuteOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmPermuteOp(op.operand, int_list_from_prop(op.dims), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ViewAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenViewOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmViewOp(op.operand, int_list_from_prop(op.shape), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ReshapeAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenReshapeOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmReshapeOp(op.operand, int_list_from_prop(op.shape), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ContiguousAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenContiguousOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmContiguousOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class FlattenAsCinmPattern(RewritePattern):
    """gen.flatten -> cinm.reshape to the shape from start_dim / end_dim. Invalid
    bounds leave the op unmatched."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenFlattenOp, rewriter: PatternRewriter) -> None:
        in_shape = shape_of(op.operand.type)
        rank = len(in_shape)
        start_dim = int_from_prop(op.start_dim)
        end_dim = int_from_prop(op.end_dim)
        if end_dim < 0:
            end_dim += rank
        if start_dim < 0:
            start_dim += rank
        if start_dim < 0 or end_dim >= rank or start_dim > end_dim:
            return
        flat = 1
        for d in in_shape[start_dim : end_dim + 1]:
            flat *= d
        out_shape = in_shape[:start_dim] + [flat] + in_shape[end_dim + 1 :]
        y = nd.CinmReshapeOp(op.operand, out_shape, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class UnsqueezeAsCinmPattern(RewritePattern):
    """gen.unsqueeze -> cinm.reshape to the result type's shape."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenUnsqueezeOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmReshapeOp(op.operand, shape_of(op.result.type), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ExpandAsCinmPattern(RewritePattern):
    """gen.broadcast -> cinm.broadcast to the result type's shape (not the shape prop,
    which can hold -1)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenBroadcastOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmBroadcastOp(op.operand, shape_of(op.result.type), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class RowAbsmaxAsCinmPattern(RewritePattern):
    """gen.row_absmax -> cinm.row_absmax."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenRowAbsmaxOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmRowAbsmaxOp(op.operand, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class QuantizePerTokenAsCinmPattern(RewritePattern):
    """gen.quantize_per_token -> cinm.quantize_per_token."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenQuantizePerTokenOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmQuantizePerTokenOp(op.operand, op.row_max, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class DequantizePerTokenAsCinmPattern(RewritePattern):
    """gen.dequantize_per_token -> cinm.dequantize_per_token (col_max operand or number)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenDequantizePerTokenOp, rewriter: PatternRewriter) -> None:
        col = op.col_max if op.col_max is not None else float_from_prop(op.col_max_scalar)
        y = nd.CinmDequantizePerTokenOp(op.operand, op.row_max, col, op.result.type, bias=op.bias)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class WhereAsCinmPattern(RewritePattern):
    """gen.where -> cinm.where (other operand or number)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenWhereOp, rewriter: PatternRewriter) -> None:
        other = op.other if op.other is not None else float_from_prop(op.other_scalar)
        y = nd.CinmWhereOp(op.cond, op.operand, other, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class QuantizePerTensorAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenQuantizePerTensorOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmQuantizePerTensorOp(op.operand, float_from_prop(op.scale), int_from_prop(op.zero_point), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class DequantizeAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenDequantizeOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmDequantizeOp(op.operand, float_from_prop(op.scale), int_from_prop(op.zero_point), op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


def _weight_type(op) -> IntegerType:
    """The stored weight's element type: signed, weight_bits wide (si8 for a qint8 weight)."""
    return IntegerType(int_from_prop(op.weight_bits), Signedness.SIGNED)


class QuantizedLinearAsCinmPattern(RewritePattern):
    """gen.quantized_linear -> const.ref(weight, signed weight_bits) [out, in] ->
    cinm.transpose -> cinm.quantized_matmul(x, w^T), with the float bias if any. The
    weight is signed (si8 for qint8) while the activation is ui8."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenQuantizedLinearOp, rewriter: PatternRewriter) -> None:
        x = op.operand
        in_features = shape_of(x.type)[-1]
        out_features = shape_of(op.result.type)[-1]
        target = op.module_target.data
        group_rows = int_from_prop(op.group_rows)
        w_ty = _weight_type(op)

        weight = ConstRefOp(op.weight_name.data, TensorType(w_ty, [out_features, in_features]))
        weight_t = nd.CinmTransposeOp(weight.result, 0, 1, TensorType(w_ty, [in_features, out_features]))
        new_ops = [weight, weight_t]

        bias_value = None
        if op.bias_name is not None:
            bias = ConstRefOp(op.bias_name.data, TensorType(f32, [out_features]))
            new_ops.append(bias)
            bias_value = bias.result

        if op.output_scale is None:
            # raw form: the int8 x int8 matmul alone, an int32 result; or the
            # group-rescaled raw form: the groups' accumulators x their factors, summed
            scale_name = None if op.combined_scale_name is None else op.combined_scale_name.data
            matmul = nd.CinmQuantizedMatmulOp(x, weight_t.result, group_rows, None, None, None, op.result.type,
                                              combined_scale_name=scale_name)
        else:
            per_channel = op.combined_scale_name is not None
            matmul = nd.CinmQuantizedMatmulOp(
                x, weight_t.result, group_rows, None if per_channel else float_from_prop(op.combined_scale),
                float_from_prop(op.output_scale), int_from_prop(op.output_zero_point),
                op.result.type, bias=bias_value,
                combined_scale_name=op.combined_scale_name.data if per_channel else None,
            )
        matmul.attributes["logical_matmul_name"] = StringAttr(target)
        matmul.attributes["logical_matmul_source_name"] = StringAttr(target)
        matmul.attributes["logical_matmul_source_op"] = StringAttr("call_module")
        copy_provenance(op, matmul)
        new_ops.append(matmul)

        rewriter.replace_matched_op(new_ops, new_results=matmul.results)


class QuantizedConv2dAsCinmPattern(RewritePattern):
    """gen.quantized_conv2d -> the int8 matmul of gen.quantized_linear over im2col
    patches (the array only does matmuls):
      cinm.unfold      x [N, C, H, W] -> [N, C*kH*kW, L], L = H_out * W_out
      cinm.transpose   -> [N, L, C*kH*kW]: one row per output position
      const.ref        weight (signed weight_bits) as [OC, C*kH*kW] -> cinm.transpose -> [C*kH*kW, OC]
      cinm.quantized_matmul   rows x weight^T, combined scale, bias -> [N, L, OC]
      cinm.transpose + cinm.reshape   -> [N, OC, H_out, W_out]
    With groups g > 1 the matmul is grouped (one weight block per group, the grouped
    stationary [g, v, k] of xdsl_hardware): v = C/g * kH*kW inputs, k = OC/g outputs
      cinm.reshape + 2 cinm.transpose   patches -> [N, L, g, v]
      const.ref        weight as [g, k, v] -> cinm.transpose -> [g, v, k]
      cinm.quantized_matmul   -> [N, L, g, k] (bias [g, k])
      cinm.reshape + cinm.transpose + cinm.reshape   -> [N, OC, H_out, W_out]"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenQuantizedConv2dOp, rewriter: PatternRewriter) -> None:
        x = op.operand
        x_ty = x.type.element_type
        y_ty = op.result.type.element_type
        n, channels = shape_of(x.type)[:2]
        out_shape = shape_of(op.result.type)
        n_out, out_channels, h_out, w_out = out_shape
        if n_out != n:
            raise ValueError(f"gen.quantized_conv2d {op.module_target.data}: batch {n} in, {n_out} out")
        kh, kw = int_list_from_prop(op.kernel_size)
        patch = channels * kh * kw
        positions = h_out * w_out
        target = op.module_target.data

        groups = int_from_prop(op.groups)
        if channels % groups or out_channels % groups:
            raise ValueError(f"gen.quantized_conv2d {target}: {channels} -> {out_channels} channels in {groups} groups")
        v = patch // groups            # inputs per group: C/g * kH*kW
        k = out_channels // groups     # outputs per group
        group_rows = int_from_prop(op.group_rows)
        w_ty = _weight_type(op)

        unfolded = nd.CinmUnfoldOp(
            x, [kh, kw], TensorType(x_ty, [n, patch, positions]),
            dilation=int_list_from_prop(op.dilation), padding=int_list_from_prop(op.padding),
            stride=int_list_from_prop(op.stride),
        )
        new_ops = [unfolded]
        if groups == 1:
            rows = nd.CinmTransposeOp(unfolded.result, 1, 2, TensorType(x_ty, [n, positions, patch]))
            weight = ConstRefOp(op.weight_name.data, TensorType(w_ty, [out_channels, patch]))
            weight_t = nd.CinmTransposeOp(weight.result, 0, 1, TensorType(w_ty, [patch, out_channels]))
            new_ops += [rows, weight, weight_t]
            channel_shape = [out_channels]
        else:
            # unfold orders each position's patch channel-major, so group i is the
            # contiguous v-wide slice [i*v, (i+1)*v)
            split = nd.CinmReshapeOp(unfolded.result, [n, groups, v, positions], TensorType(x_ty, [n, groups, v, positions]))
            pos_second = nd.CinmTransposeOp(split.result, 1, 3, TensorType(x_ty, [n, positions, v, groups]))
            rows = nd.CinmTransposeOp(pos_second.result, 2, 3, TensorType(x_ty, [n, positions, groups, v]))
            weight = ConstRefOp(op.weight_name.data, TensorType(w_ty, [groups, k, v]))
            weight_t = nd.CinmTransposeOp(weight.result, 1, 2, TensorType(w_ty, [groups, v, k]))
            new_ops += [split, pos_second, rows, weight, weight_t]
            channel_shape = [groups, k]

        bias_value = None
        if op.bias_name is not None:
            bias = ConstRefOp(op.bias_name.data, TensorType(f32, channel_shape))
            new_ops.append(bias)
            bias_value = bias.result

        per_channel = op.combined_scale_name is not None
        matmul = nd.CinmQuantizedMatmulOp(
            rows.result, weight_t.result, group_rows, None if per_channel else float_from_prop(op.combined_scale),
            float_from_prop(op.output_scale), int_from_prop(op.output_zero_point),
            TensorType(y_ty, [n, positions, *channel_shape]), bias=bias_value,
            combined_scale_name=op.combined_scale_name.data if per_channel else None,
        )
        matmul.attributes["logical_matmul_name"] = StringAttr(target)
        matmul.attributes["logical_matmul_source_name"] = StringAttr(target)
        matmul.attributes["logical_matmul_source_op"] = StringAttr("call_module")
        new_ops.append(matmul)
        flat = matmul.result
        if groups != 1:
            merged = nd.CinmReshapeOp(matmul.result, [n, positions, out_channels],
                                      TensorType(y_ty, [n, positions, out_channels]))
            new_ops.append(merged)
            flat = merged.result
        channels_first = nd.CinmTransposeOp(flat, 1, 2, TensorType(y_ty, [n, out_channels, positions]))
        y = nd.CinmReshapeOp(channels_first.result, out_shape, op.result.type)
        new_ops += [channels_first, y]
        for new_op in new_ops:
            copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_ops, new_results=y.results)


class QuantizedMatmulAsCinmPattern(RewritePattern):
    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenQuantizedMatmulOp, rewriter: PatternRewriter) -> None:
        # two activations, each with one scale: one group over the whole reduction dim
        group_rows = shape_of(op.lhs.type)[-1]
        if op.output_scale is None:
            # raw form: the int8 x int8 matmul alone, an int32 result
            y = nd.CinmQuantizedMatmulOp(op.lhs, op.rhs, group_rows, None, None, None, op.result.type)
        else:
            y = nd.CinmQuantizedMatmulOp(
                op.lhs, op.rhs, group_rows, float_from_prop(op.combined_scale),
                float_from_prop(op.output_scale), int_from_prop(op.output_zero_point), op.result.type,
            )
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class CatAsCinmPattern(RewritePattern):
    """gen.cat -> cinm.cat (tensors, dim)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenCatOp, rewriter: PatternRewriter) -> None:
        y = nd.CinmCatOp(list(op.tensors), op.result.type, dim=int_from_prop(op.dim))
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class SliceAsCinmPattern(RewritePattern):
    """gen.slice -> common.slice (same fields). common.slice is shared by every stage,
    so later passes handle one slice op."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenSliceOp, rewriter: PatternRewriter) -> None:
        y = CommonSliceOp(
            op.operand, int_list_from_prop(op.starts), int_list_from_prop(op.stops),
            int_list_from_prop(op.steps), op.result.type,
        )
        copy_provenance(op, y)
        rewriter.replace_matched_op(y)


class ChunkAsCinmPattern(RewritePattern):
    """gen.chunk -> one common.slice per equal piece along dim. Only a dim size
    divisible by chunks; otherwise left unmatched."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: gd.GenChunkOp, rewriter: PatternRewriter) -> None:
        in_shape = shape_of(op.operand.type)
        rank = len(in_shape)
        dim = int_from_prop(op.dim)
        if dim < 0:
            dim += rank
        chunks = int_from_prop(op.chunks)
        size = in_shape[dim]
        if size % chunks != 0:
            return
        chunk_size = size // chunks

        new_ops = []
        for i in range(chunks):
            starts = [SLICE_NONE] * rank
            stops = [SLICE_NONE] * rank
            steps = [SLICE_NONE] * rank
            starts[dim] = i * chunk_size
            stops[dim] = (i + 1) * chunk_size
            new_ops.append(CommonSliceOp(op.operand, starts, stops, steps, op.results[i].type))
        for new_op in new_ops:
            copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_ops, new_results=[o.result for o in new_ops])


def run_gen_to_cinm_canon(module) -> None:
    """Every gen op to its cinm op, in one greedy pass (the patterns match disjoint op
    types). apply_recursively=False: a new cinm.vec_add / vec_div is not matched again
    by Add/DivAsCinmPattern in the same pass."""
    PatternRewriteWalker(
        GreedyRewritePatternApplier(
            [
                QuantizePerTensorAsCinmPattern(),
                RowAbsmaxAsCinmPattern(),
                QuantizePerTokenAsCinmPattern(),
                DequantizePerTokenAsCinmPattern(),
                WhereAsCinmPattern(),
                DequantizeAsCinmPattern(),
                QuantizedLinearAsCinmPattern(),
                QuantizedConv2dAsCinmPattern(),
                QuantizedMatmulAsCinmPattern(),
                SoftmaxAsCinmPattern(),
                GeluAsCinmPattern(),
                SiluAsCinmPattern(),
                SiluModuleAsCinmPattern(),
                GeluModuleAsCinmPattern(),
                EluAsCinmPattern(),
                SqrtAsCinmPattern(),
                TransposeAsCinmPattern(),
                AddAsCinmPattern(),
                MulAsCinmPattern(),
                DivAsCinmPattern(),
                PermuteAsCinmPattern(),
                ViewAsCinmPattern(),
                ReshapeAsCinmPattern(),
                ContiguousAsCinmPattern(),
                FlattenAsCinmPattern(),
                UnsqueezeAsCinmPattern(),
                ExpandAsCinmPattern(),
                CatAsCinmPattern(),
                SliceAsCinmPattern(),
                ChunkAsCinmPattern(),
                MeanAsCinmPattern(),
                ArgmaxAsCinmPattern(),
                MaxAsCinmPattern(),
                ScatterAsCinmPattern(),
            ]
        ),
        apply_recursively=False,
    ).rewrite_module(module)
    require_lowered(module, "gen.", "gen -> cinm")


def run_rewrite_cinm_applicable_canon(module) -> None:
    """The module-level rewrites, in order: RMSNorm before the generic norm (it
    consumes a gen.norm)."""
    run_layer_norm_canon(module)
    run_group_like_norm_canon(module)
    run_rms_norm_canon(module)
    run_norm_canon(module)
    run_embedding_canon(module)
    run_relu_canon(module)
    run_leaky_relu_canon(module)
    run_max_pool2d_canon(module)
    run_adaptive_avg_pool2d_canon(module)
    run_upsample_canon(module)
