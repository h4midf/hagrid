"""The cinm -> cim lowering, in place (the module mixes both dialects while it runs):

lower_cinm_to_cim                     every cinm op to its cim op(s): the 1:1 swaps
                                      in _DIRECT_REPLACEMENTS plus the patterns below
convert_stationary_to_cim_stationary  every const.ref (weight or LUT table) wrapped
                                      in cim.save_stationary
"""

from xdsl.dialects.builtin import ModuleOp, TensorType, f32, i32
from xdsl.pattern_rewriter import (
    GreedyRewritePatternApplier,
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)
from xdsl.rewriter import InsertPoint

from compiler.dialects import cim_dialect as cd
from compiler.dialects import cinm_dialect as nd
from compiler.dialects.attr_helpers import float_from_prop, int_from_prop, int_list_from_prop
from compiler.dialects.common_dialect import ConstRefOp, ui8
from compiler.dialects.rewrite_helpers import (
    copy_provenance, dropped_last_dim, require_lowered, scalar_rhs_of, shape_of,
)


class _DirectReplace(RewritePattern):
    """A 1:1 swap of a `cinm_cls` op for the cim op `build(op)` returns."""

    def __init__(self, cinm_cls, build):
        self.cinm_cls = cinm_cls
        self.build = build

    def match_and_rewrite(self, op, rewriter: PatternRewriter) -> None:
        if not isinstance(op, self.cinm_cls):
            return
        new_op = self.build(op)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


class LutOpAsLutSearchPattern(RewritePattern):
    """cinm.LUTOp with a 3D table -> cim.lut_search_vector (no affine index map);
    otherwise cim.lut_search with the LUT's alpha / beta."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmLUTOp, rewriter: PatternRewriter) -> None:
        if len(shape_of(op.table.type)) == 3:
            new_op = cd.CimLutSearchVectorOp(op.table, op.index, op.result.type)
        else:
            new_op = cd.CimLutSearchOp(
                op.table, op.index, float_from_prop(op.alpha), float_from_prop(op.beta), op.result.type,
            )
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


class LutOpFusedAsLutSearchPattern(RewritePattern):
    """cinm.LUTOpFused (the variadic LUT) with one index arg -> cim.lut_search /
    lut_search_vector with the identity index map. Raises for any other arg count."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmLUTOpFused, rewriter: PatternRewriter) -> None:
        args = list(op.args)
        if len(args) != 1:
            raise NotImplementedError(f"{op.name}: a fused LUT with {len(args)} index args has no cim lowering")
        index = args[0]
        if len(shape_of(op.table.type)) == 3:
            new_op = cd.CimLutSearchVectorOp(op.table, index, op.result.type)
        else:
            new_op = cd.CimLutSearchOp(op.table, index, 1.0, 0.0, op.result.type)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


class ReduceVecAsReduceVecBroadcastPattern(RewritePattern):
    """cinm.reduceVec (sum over the last axis, keepdim) -> cim.reduce_vec (drops the
    axis) + cim.broadcast back to the keepdim shape."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmReduceVecOp, rewriter: PatternRewriter) -> None:
        reduced_type = dropped_last_dim(op.operand.type)
        red = cd.CimReduceVecOp(op.operand, -1, reduced_type)
        copy_provenance(op, red)
        out = cd.CimBroadcastOp(red.result, shape_of(op.result.type), op.result.type)
        copy_provenance(op, out)
        rewriter.replace_matched_op([red, out], new_results=out.results)


class DotProductAsDecomposePattern(RewritePattern):
    """cinm.dotProduct(a, b) = sum(a*b, last axis, keepdim) -> cim.vec_el_mul + the
    reduce + broadcast of ReduceVecAsReduceVecBroadcastPattern."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmDotProductOp, rewriter: PatternRewriter) -> None:
        prod = cd.CimVecElMulOp(op.lhs, op.rhs, op.lhs.type)
        copy_provenance(op, prod)
        reduced_type = dropped_last_dim(prod.result.type)
        red = cd.CimReduceVecOp(prod.result, -1, reduced_type)
        copy_provenance(op, red)
        out = cd.CimBroadcastOp(red.result, shape_of(op.result.type), op.result.type)
        copy_provenance(op, out)
        rewriter.replace_matched_op([prod, red, out], new_results=out.results)


class NegAsVecElMulPattern(RewritePattern):
    """cinm.neg(x) -> cim.vec_el_mul(x, -1.0)."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmNegOp, rewriter: PatternRewriter) -> None:
        new_op = cd.CimVecElMulOp(op.operand, -1.0, op.result.type)
        copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_op)


class StationaryConversionPattern(RewritePattern):
    """Wraps a const.ref (weight or LUT table) in cim.save_stationary (stored in the
    RMU). A sole following transpose, reshape, or reshape-then-transpose is fused into
    the save_stationary, so the stored layout already has it; otherwise every use of
    the const is redirected to the wrapped value."""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: ConstRefOp, rewriter: PatternRewriter) -> None:
        uses = list(op.result.uses)
        single_user = uses[0].operation if len(uses) == 1 else None

        if isinstance(single_user, cd.CimTransposeOp):
            self._fuse_transpose(op, single_user, rewriter)
        elif isinstance(single_user, cd.CimReshapeOp):
            reshape_uses = list(single_user.result.uses)
            reshape_only_user = reshape_uses[0].operation if len(reshape_uses) == 1 else None
            if isinstance(reshape_only_user, cd.CimTransposeOp):
                self._fuse_reshape_then_transpose(op, single_user, reshape_only_user, rewriter)
            else:
                self._fuse_reshape(op, single_user, rewriter)
        else:
            self._plain_wrap(op, rewriter)

    @staticmethod
    def _fuse_transpose(const: ConstRefOp, tr: "cd.CimTransposeOp", rewriter: PatternRewriter) -> None:
        stat = cd.CimSaveStationaryOp(const.result, tr.result.type, transposed=True)
        copy_provenance(tr, stat)
        rewriter.insert_op(stat, InsertPoint.before(tr))
        rewriter.replace_op(tr, [], new_results=[stat.result])

    @staticmethod
    def _fuse_reshape(const: ConstRefOp, rs: "cd.CimReshapeOp", rewriter: PatternRewriter) -> None:
        new_shape = int_list_from_prop(rs.shape)
        stat = cd.CimSaveStationaryOp(const.result, rs.result.type, out_shape=new_shape)
        copy_provenance(rs, stat)
        rewriter.insert_op(stat, InsertPoint.before(rs))
        rewriter.replace_op(rs, [], new_results=[stat.result])

    @staticmethod
    def _fuse_reshape_then_transpose(
        const: ConstRefOp, rs: "cd.CimReshapeOp", tr: "cd.CimTransposeOp", rewriter: PatternRewriter
    ) -> None:
        reshaped_shape = int_list_from_prop(rs.shape)
        stat = cd.CimSaveStationaryOp(const.result, tr.result.type, transposed=True, out_shape=reshaped_shape)
        copy_provenance(tr, stat)
        rewriter.insert_op(stat, InsertPoint.before(tr))
        rewriter.replace_op(tr, [], new_results=[stat.result])
        rewriter.erase_op(rs)

    @staticmethod
    def _plain_wrap(const: ConstRefOp, rewriter: PatternRewriter) -> None:
        stat = cd.CimSaveStationaryOp(const.result, const.result.type)
        copy_provenance(const, stat)
        rewriter.insert_op(stat, InsertPoint.after(const))
        rewriter.replace_uses_with_if(const.result, stat.result, lambda use: use.operation is not stat)


def _merge_same_name_const_refs(module: ModuleOp) -> None:
    """A const.ref's name is its stored buffer, so const.refs with the same name (e.g.
    the identity table built by both the quantize LUT and the requantize step) are one:
    every use goes to the first, the rest are erased. Raises if their types differ."""
    first_by_name: dict[str, ConstRefOp] = {}
    duplicates = []
    for op in module.walk():
        if not isinstance(op, ConstRefOp):
            continue
        name = op.sym_name.data
        first = first_by_name.get(name)
        if first is None:
            first_by_name[name] = op
            continue
        if op.result.type != first.result.type:
            raise ValueError(f"const.ref {name!r}: two types {first.result.type} and {op.result.type} "
                             "for one buffer")
        duplicates.append((op, first))
    for dup, first in duplicates:
        dup.result.replace_by(first.result)
        dup.detach()
        dup.erase()


def convert_stationary_to_cim_stationary(module: ModuleOp) -> None:
    """In place: merges const.refs naming the same buffer, then wraps every const.ref in
    cim.save_stationary (StationaryConversionPattern). Runs after lower_cinm_to_cim."""
    _merge_same_name_const_refs(module)
    PatternRewriteWalker(StationaryConversionPattern(), apply_recursively=False).rewrite_module(module)


_DIRECT_REPLACEMENTS = [
    (nd.CinmEmbeddingOp, lambda op: cd.CimEmbeddingOp(op.table, op.index, op.result.type)),
    (nd.CinmVecElMulOp, lambda op: cd.CimVecElMulOp(op.lhs, scalar_rhs_of(op), op.result.type)),
    (nd.CinmVecAddOp, lambda op: cd.CimVecAddOp(op.lhs, scalar_rhs_of(op), op.result.type)),
    (nd.CinmScalarAddOp, lambda op: cd.CimVecAddOp(op.lhs, scalar_rhs_of(op), op.result.type)),
    (nd.CinmTransposeOp, lambda op: cd.CimTransposeOp(
        op.operand, int_from_prop(op.dim0), int_from_prop(op.dim1), op.result.type
    )),
    (nd.CinmPermuteOp, lambda op: cd.CimPermuteOp(op.operand, int_list_from_prop(op.dims), op.result.type)),
    (nd.CinmViewOp, lambda op: cd.CimViewOp(op.operand, int_list_from_prop(op.shape), op.result.type)),
    (nd.CinmReshapeOp, lambda op: cd.CimReshapeOp(op.operand, int_list_from_prop(op.shape), op.result.type)),
    (nd.CinmContiguousOp, lambda op: cd.CimContiguousOp(op.operand, op.result.type)),
    (nd.CinmBroadcastOp, lambda op: cd.CimBroadcastOp(op.operand, int_list_from_prop(op.shape), op.result.type)),
    (nd.CinmUnfoldOp, lambda op: cd.CimUnfoldOp(
        op.operand, int_list_from_prop(op.kernel_size), op.result.type,
        dilation=int_list_from_prop(op.dilation), padding=int_list_from_prop(op.padding),
        stride=int_list_from_prop(op.stride),
    )),
    (nd.CinmMaxOp, lambda op: cd.CimMaxOp(
        op.operand, op.result.type, reduce_dim=int_from_prop(op.reduce_dim), k=int_from_prop(op.k),
    )),
    (nd.CinmArgmaxOp, lambda op: cd.CimArgmaxOp(
        op.operand, op.result.type, reduce_dim=int_from_prop(op.reduce_dim), k=int_from_prop(op.k),
    )),
    (nd.CinmCatOp, lambda op: cd.CimCatOp(list(op.tensors), op.result.type, dim=int_from_prop(op.dim))),
    (nd.CinmScatterOp, lambda op: cd.CimScatterOp(op.index, op.src, op.result.type, dim=int_from_prop(op.dim))),
    # The activations, norms and quantize/dequantize below reach this stage only when
    # their impl is a circuit (a "lut" impl was already rewritten into a LUT op by
    # compiler/ops/<kind>.py). A circuit's choice is read back from the attributes that
    # handler stamped on the op: "variant" (gelu, softmax, layer_norm), and the
    # (sh, m, zp, mantissa_width) encoding (quantize, dequantize).
    (nd.CinmReluOp, lambda op: cd.CimReluOp(op.operand, op.result.type)),
    (nd.CinmRsqrtOp, lambda op: cd.CimRsqrtOp(op.operand, op.result.type)),
    (nd.CinmGeluOp, lambda op: cd.CimGeluOp(op.operand, op.result.type, variant=op.attributes["variant"].data)),
    (nd.CinmSoftmaxOp, lambda op: cd.CimSoftmaxOp(
        op.operand, op.result.type, variant=op.attributes["variant"].data, dim=int_from_prop(op.dim),
    )),
    # silu's `inplace` property is dropped: nothing reads it
    (nd.CinmSiluOp, lambda op: cd.CimSiluOp(op.operand, op.result.type)),
    # alpha is carried over: xdsl_timing.py's _spec_elu picks the characterized
    # circuit by it
    (nd.CinmEluOp, lambda op: cd.CimEluOp(op.operand, op.result.type, alpha=float_from_prop(op.alpha))),
    # negative_slope is carried over, same as CinmEluOp's alpha
    (nd.CinmLeakyReluOp, lambda op: cd.CimLeakyReluOp(
        op.operand, op.result.type,
        negative_slope=float_from_prop(op.negative_slope),
    )),
    # hidden_size picks the characterized circuit (with variant)
    (nd.CinmLayerNormOp, lambda op: cd.CimLayerNormOp(
        op.operand, op.result.type, hidden_size=int_from_prop(op.hidden_size), eps=float_from_prop(op.eps),
        variant=op.attributes["variant"].data,
        weight_name=op.weight_name.data if op.weight_name is not None else None,
        bias_name=op.bias_name.data if op.bias_name is not None else None,
    )),
    # RMSNorm has no bias
    (nd.CinmRMSNormOp, lambda op: cd.CimRMSNormOp(
        op.operand, op.result.type, hidden_size=int_from_prop(op.hidden_size), eps=float_from_prop(op.eps),
        weight_name=op.weight_name.data if op.weight_name is not None else None,
    )),
    # m / mantissa_width exist only for a mant_* impl, zp only for a *_zp impl
    (nd.CinmQuantizePerTensorOp, lambda op: cd.CimQuantizeOp(
        op.operand, op.result.type, variant=op.attributes["variant"].data, sh=op.attributes["sh"].value.data,
        m=op.attributes["m"].value.data if "m" in op.attributes else None,
        zp=op.attributes["zp"].value.data if "zp" in op.attributes else None,
        mantissa_width=op.attributes["mantissa_width"].value.data if "mantissa_width" in op.attributes else None,
    )),
    (nd.CinmDequantizeOp, lambda op: cd.CimDequantizeOp(
        op.operand, op.result.type, variant=op.attributes["variant"].data, sh=op.attributes["sh"].value.data,
        m=op.attributes["m"].value.data if "m" in op.attributes else None,
        zp=op.attributes["zp"].value.data if "zp" in op.attributes else None,
        mantissa_width=op.attributes["mantissa_width"].value.data if "mantissa_width" in op.attributes else None,
    )),
    # per-token int8 quantisation (kernels/gpt2.py): custom_op_library/token_quant
    (nd.CinmRowAbsmaxOp, lambda op: cd.CimRowAbsmaxOp(op.operand, op.result.type)),
    (nd.CinmQuantizePerTokenOp, lambda op: cd.CimQuantizePerTokenOp(op.operand, op.row_max, op.result.type)),
    # where: custom_op_library/vec_select, which takes one fill word
    (nd.CinmWhereOp, lambda op: cd.CimVecSelectOp(op.cond, op.operand, _where_fill(op), op.result.type)),
]


def _where_fill(op: nd.CinmWhereOp) -> float:
    """cinm.where's other as vec_select's fill word: one number only."""
    if op.other_scalar is None:
        raise NotImplementedError("cinm.where with a tensor `other`: vec_select takes one fill word, "
                                  "an elementwise choice between two tensors is not modeled")
    return float_from_prop(op.other_scalar)


# 1 / 127^2: the two maxima of a per-token dequantisation are scales * 127
_INV_INT8_MAX_SQ = 1.0 / (127 * 127)


class DequantizePerTokenAsCimPattern(RewritePattern):
    """cinm.dequantize_per_token -> acc * (row_max / 127) * (col_max / 127) (+ bias) on
    the vector units (custom_op_library vec_el_mul / vec_add), no op of its own:
      cim.vec_el_mul   acc (i32) * col -> f32
      cim.vec_el_mul   * row_max (one per row, broadcast over the row)
      cim.vec_el_mul   * 1 / 127^2, only when col is computed at run time (Q.K^T's K
                       maxima); a stored col (a Linear's per-channel maxima) or a number
                       (P.V's static V maximum) has it folded in offline / at compile time
      cim.vec_add      + bias (a Linear's), if any"""

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmDequantizePerTokenOp, rewriter: PatternRewriter) -> None:
        f32_type = TensorType(f32, shape_of(op.result.type))
        new_ops = []
        if op.col_max_scalar is not None:
            col = float_from_prop(op.col_max_scalar) * _INV_INT8_MAX_SQ
            fold_done = True
        else:
            col = op.col_max
            # a stored col (const.ref) holds col_max / 127^2, folded offline
            fold_done = isinstance(op.col_max.owner, ConstRefOp)
        by_col = cd.CimVecElMulOp(op.operand, col, f32_type)
        by_row = cd.CimVecElMulOp(by_col.result, op.row_max, f32_type)
        new_ops += [by_col, by_row]
        out = by_row
        if not fold_done:
            out = cd.CimVecElMulOp(out.result, _INV_INT8_MAX_SQ, f32_type)
            new_ops.append(out)
        if op.bias is not None:
            out = cd.CimVecAddOp(out.result, op.bias, f32_type)
            new_ops.append(out)
        for new_op in new_ops:
            copy_provenance(op, new_op)
        rewriter.replace_matched_op(new_ops, new_results=out.results)


class QuantizedMatmulAsCimPattern(RewritePattern):
    """cinm.quantized_matmul -> the int8 requantize sequence (the raw form: the
    cim.matvec alone):
      cim.matvec        int8 x int8, i32 accumulator
      cim.vec_el_mul    * combined_scale (input scale * weight scale) -> real units:
                        one float for a per-tensor weight, or a stored [out_features]
                        const (combined_scale_name) for a per-channel weight
      cim.vec_add       + bias (quantized Linear only; quantized.matmul has none)
      cim.quantize      requantize on the custom_op_library quantize circuit, when
                        compiler/ops/requantize.py stamped its encoding (a circuit
                        requantize_impl); otherwise
      cim.lut_search    requantize: idx = round(x / output_scale + zero_point) into the
                        256-entry identity table, the same lookup as quantize "lut"
    Split here, not kept atomic, so tiling (tile_matmuls, after this) can sum the
    per-tile i32 accumulators before the rescale. One identity table is shared by
    every quantized_matmul in the pass.
    A group-wise weight (group_rows below the reduction dim: ceil(dim / group_rows)
    groups, kernels/w3.py) is rescaled per group before the groups are summed:
      cim.matvec        one i32 partial result per group of group_rows rows,
                        [..., groups, out] (the row-tiled form of tile_matmuls)
      cim.vec_el_mul    * the [groups, out] combined scales (combined_scale_name)
      cim.reduce_vec    sum over the groups -> [..., out]
    then the bias and requantize as above."""

    def __init__(self):
        super().__init__()
        self._table: ConstRefOp | None = None

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: nd.CinmQuantizedMatmulOp, rewriter: PatternRewriter) -> None:
        out_shape = shape_of(op.result.type)
        i32_type = TensorType(i32, out_shape)
        f32_type = TensorType(f32, out_shape)
        reduce_dim = shape_of(op.lhs.type)[-1]
        groups = -(-reduce_dim // int_from_prop(op.group_rows))
        if groups > 1:
            self._rewrite_group_wise(op, rewriter, groups)
            return

        matvec = cd.CimMatvecOp(op.lhs, op.rhs, i32_type)
        copy_provenance(op, matvec)
        if op.output_scale is None:
            # raw form: the cim.matvec alone, its i32 accumulator the result
            rewriter.replace_matched_op(matvec)
            return
        new_ops = [matvec]
        if op.combined_scale_name is not None:
            # per-channel weight: one combined scale per output channel, a stored
            # [out_features] vector (wrapped in save_stationary like the bias); a grouped
            # matmul (rhs [g, v, k]) has its channels as [g, k], the output's last two dims
            channel_shape = out_shape[-2:] if len(shape_of(op.rhs.type)) == 3 else [out_shape[-1]]
            scale_vec = ConstRefOp(op.combined_scale_name.data, TensorType(f32, channel_shape))
            new_ops.append(scale_vec)
            scaled = cd.CimVecElMulOp(matvec.result, scale_vec.result, f32_type)
        else:
            scaled = cd.CimVecElMulOp(matvec.result, float_from_prop(op.combined_scale), f32_type)
        copy_provenance(op, scaled)
        new_ops.append(scaled)
        self._finish(op, rewriter, new_ops, scaled.result)

    def _rewrite_group_wise(self, op: nd.CinmQuantizedMatmulOp, rewriter: PatternRewriter, groups: int) -> None:
        if op.combined_scale_name is None:
            raise NotImplementedError("cinm.quantized_matmul: a group-wise weight needs its [groups, out] "
                                      "combined scales (combined_scale_name)")
        if len(shape_of(op.rhs.type)) != 2:
            raise NotImplementedError(f"cinm.quantized_matmul: a grouped matmul (rhs {shape_of(op.rhs.type)}) "
                                      f"with {groups} weight groups per conv group is not supported")
        out_shape = shape_of(op.result.type)
        pre_shape = list(out_shape[:-1]) + [groups, out_shape[-1]]
        matvec = cd.CimMatvecOp(op.lhs, op.rhs, TensorType(i32, pre_shape), out_shape=pre_shape)
        scale_vec = ConstRefOp(op.combined_scale_name.data, TensorType(f32, [groups, out_shape[-1]]))
        scaled = cd.CimVecElMulOp(matvec.result, scale_vec.result, TensorType(f32, pre_shape))
        summed = cd.CimReduceVecOp(scaled.result, -2, TensorType(f32, out_shape), out_shape=tuple(out_shape))
        for new_op in (matvec, scaled, summed):
            copy_provenance(op, new_op)
        if op.output_scale is None:
            # group-rescaled raw form: the summed groups are the result, dequantized by a
            # separate op (cinm.dequantize_per_token)
            rewriter.replace_matched_op([matvec, scale_vec, scaled, summed], new_results=summed.results)
            return
        self._finish(op, rewriter, [matvec, scale_vec, scaled, summed], summed.result)

    def _finish(self, op: nd.CinmQuantizedMatmulOp, rewriter: PatternRewriter, new_ops: list,
                scaled) -> None:
        """+ bias, then the requantize, after the rescale to real units (`scaled`)."""
        f32_type = TensorType(f32, shape_of(op.result.type))
        pre_requant = scaled
        if op.bias is not None:
            added = cd.CimVecAddOp(scaled, op.bias, f32_type)
            copy_provenance(op, added)
            new_ops.append(added)
            pre_requant = added.result

        if "requant_variant" in op.attributes:
            # a circuit requantize (compiler/ops/requantize.py stamped its encoding):
            # the custom_op_library quantize circuit, a CMOS unit, not an array lookup
            attrs = op.attributes
            requant = cd.CimQuantizeOp(
                pre_requant, op.result.type, variant=attrs["requant_variant"].data,
                sh=attrs["requant_sh"].value.data,
                m=attrs["requant_m"].value.data if "requant_m" in attrs else None,
                zp=attrs["requant_zp"].value.data if "requant_zp" in attrs else None,
                mantissa_width=attrs["requant_mantissa_width"].value.data if "requant_mantissa_width" in attrs else None,
            )
            requant.attributes["accel_data_width"] = attrs["requant_data_width"]
        else:
            if self._table is None:
                self._table = ConstRefOp("quantize_identity_lut_n256", TensorType(ui8, [256]))
                new_ops.insert(0, self._table)
            alpha = 1.0 / float_from_prop(op.output_scale)
            beta = float(int_from_prop(op.output_zero_point))
            requant = cd.CimLutSearchOp(self._table.result, pre_requant, alpha, beta, op.result.type)
        new_ops.append(requant)

        copy_provenance(op, requant)
        rewriter.replace_matched_op(new_ops, new_results=requant.results)


def cinm_to_cim_patterns() -> list[RewritePattern]:
    patterns: list[RewritePattern] = [_DirectReplace(cinm_cls, build) for cinm_cls, build in _DIRECT_REPLACEMENTS]
    patterns += [
        LutOpAsLutSearchPattern(),
        LutOpFusedAsLutSearchPattern(),
        ReduceVecAsReduceVecBroadcastPattern(),
        DotProductAsDecomposePattern(),
        NegAsVecElMulPattern(),
        QuantizedMatmulAsCimPattern(),
        DequantizePerTokenAsCimPattern(),
    ]
    return patterns


def lower_cinm_to_cim(module: ModuleOp) -> None:
    """In place: every cinm op in `module` to its cim op(s), in one greedy pass. Raises
    if any cinm op is left."""
    PatternRewriteWalker(GreedyRewritePatternApplier(cinm_to_cim_patterns())).rewrite_module(module)
    require_lowered(module, "cinm.", "cinm -> cim")
