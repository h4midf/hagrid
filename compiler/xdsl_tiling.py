"""cim -> cim rewrites after the cim lowering:

tile_matmuls                     a matvec whose reduction dim exceeds the array rows
                                 becomes a matvec with one partial result per row
                                 tile, then a reduce_vec over the tiles
run_convert_static_values_canon  a literal scalar operand of vec_el_mul / vec_add
                                 becomes a stored (1,) buffer
"""

from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.pattern_rewriter import (
    PatternRewriter,
    PatternRewriteWalker,
    RewritePattern,
    op_type_rewrite_pattern,
)

from compiler.dialects import cim_dialect as cd
from compiler.dialects.common_dialect import ConstRefOp
from compiler.dialects.rewrite_helpers import copy_provenance, shape_of


class TileMatmulPattern(RewritePattern):
    """Tiles a matvec whose lhs last dim (the reduction dim) exceeds its array rows into
    ceil(dim / rows) row tiles: stored_rhs_rows for a matvec with a stored rhs,
    dynamic_rhs_rows for one with both operands dynamic. The partial results keep the
    matvec's element type (the int8 matvec's i32 accumulator), and the reduce_vec sums
    them in it."""

    def __init__(self, stored_rhs_rows: int, dynamic_rhs_rows: int):
        self.stored_rhs_rows = stored_rhs_rows
        self.dynamic_rhs_rows = dynamic_rhs_rows

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cd.CimMatvecOp, rewriter: PatternRewriter) -> None:
        lhs_shape = shape_of(op.lhs.type)
        out_shape_orig = shape_of(op.result.type)
        h_last = lhs_shape[-1]
        stored = isinstance(op.rhs.owner, cd.CimSaveStationaryOp)
        array_rows = self.stored_rhs_rows if stored else self.dynamic_rhs_rows
        if len(out_shape_orig) == len(lhs_shape) + 1:
            # already row-tiled (a group-wise weight's per-group partials,
            # QuantizedMatmulAsCimPattern): its groups must be the array's row tiles
            if out_shape_orig[-2] != -(-h_last // array_rows):
                raise NotImplementedError(f"matvec of {h_last} rows in {out_shape_orig[-2]} weight groups: "
                                          f"the groups are not the {array_rows}-row array tiles")
            return
        if h_last <= array_rows:
            return  # fits in one array
        if len(shape_of(op.rhs.type)) == 3:
            raise NotImplementedError(f"grouped matvec with {h_last} inputs per group, more than the "
                                      f"{array_rows} array rows: tiling one group is not supported")

        tiles = -(-h_last // array_rows)  # ceil division
        lead_out = out_shape_orig[:-1]
        f = out_shape_orig[-1]
        pre_shape = list(lead_out) + [tiles, f]

        pre_type = TensorType(op.result.type.element_type, pre_shape)
        mm_tiled = cd.CimMatvecOp(op.lhs, op.rhs, pre_type, out_shape=pre_shape)
        red = cd.CimReduceVecOp(mm_tiled.result, -2, op.result.type, out_shape=tuple(out_shape_orig))
        copy_provenance(op, mm_tiled)
        copy_provenance(op, red)
        rewriter.replace_matched_op([mm_tiled, red], new_results=red.results)


def tile_matmuls(module: ModuleOp, array_rows: int, dynamic_rhs_rows: int | None = None) -> None:
    """In place: every over-wide cim.matvec into a tiled matvec + reduce_vec, at
    array_rows, or at dynamic_rhs_rows for a matvec with both operands dynamic when
    given (a dual-module target: those run in the other module).
    apply_recursively=False is required: the tiled matvec keeps the same over-wide lhs,
    so revisiting it would tile it again forever."""
    pattern = TileMatmulPattern(array_rows, array_rows if dynamic_rhs_rows is None else dynamic_rhs_rows)
    PatternRewriteWalker(pattern, apply_recursively=False).rewrite_module(module)


class StaticValueAsStationaryPattern(RewritePattern):
    """The literal scalar rhs of a cim.vec_el_mul / cim.vec_add becomes a (1,)
    const.ref (__static_val_<n>, numbered per pass) in cim.save_stationary, used as the
    rhs operand."""

    def __init__(self):
        super().__init__()
        self._counter = 0

    @op_type_rewrite_pattern
    def match_and_rewrite(self, op: cd.CimVecElMulOp | cd.CimVecAddOp, rewriter: PatternRewriter) -> None:
        if op.rhs_scalar is None:
            return
        # the scalar is stored in the op's output format (a fractional scale on an i32
        # accumulator is a fixed-point value, not an integer)
        const_ty = TensorType(op.result.type.element_type, [1])

        const = ConstRefOp(f"__static_val_{self._counter}", const_ty)
        self._counter += 1
        stat = cd.CimSaveStationaryOp(const.result, const_ty)

        y = type(op)(op.lhs, stat.result, op.result.type)
        copy_provenance(op, y)
        rewriter.replace_matched_op([const, stat, y], new_results=y.results)


def run_convert_static_values_canon(module: ModuleOp) -> None:
    PatternRewriteWalker(StaticValueAsStationaryPattern(), apply_recursively=False).rewrite_module(module)
