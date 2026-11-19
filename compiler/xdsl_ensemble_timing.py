"""Timing model of the RRAM/SRAM ensemble (Krishnan et al., TCAD 2022): EnsembleTimingMapping.

The spec propagation, pf caps and memos are TimingOnlyMapping's. Only the matvec costs
differ, by their rhs:

  matvec, stored rhs   on the RRAM macros, with its SRAM correction in lockstep (the
  (weights)            paper's ensemble: the SRAM macros compute a correction of the same
                       MVM on the same input bits, added by the shifter). The RRAM side,
                       per input row: the input vector read from the SRAM (the WMU, as
                       1F1C's), then per input bit the slower of the RRAM MAC,
                       cycles_per_mac cycles (the 8-to-1 column mux, Sec. III-D-1), and the
                       SRAM correction; every row and column tile in its own macro at
                       once, the row tiles' partial sums added by the reduce_vec after the
                       matvec. pf counts RRAM copies (one input row each). The SRAM side
                       (correction_plan) spreads its weights over enough stationary macros
                       to keep pace, at most sram_macros_per_rram_macro per RRAM macro
                       (the test chip's pairing); with fewer, it is the slower side.
  matvec, both         on the SRAM macros' PEs: the rhs (K^T, V) written into the macros,
  operands dynamic     one PE per output column, a PE's weights_per_pe rows per row tile
                       (the tiling), each rhs value in ceil(bits / weight_bits) slices on
                       their own macros, combined by the shift-and-add. Per input row: the
                       input read, then per input bit each PE adds its tile's rows,
                       cycles_per_weight cycles each (the validated macro logic,
                       periphery/rtl). setup = writing the rhs into a macro. pf counts
                       SRAM copies (one input row each).
  everything else      as on a single-module target (TimingOnlyMapping), around SRAM
                       (WMU) reads and writes; a LUT entry is an RRAM row read.
The macro operations come from EnsembleTiming (targets/target_support.py).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

from xdsl.ir import BlockArgument, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_timing import TimingOnlyMapping, matvec_recipe
from targets.target_loader import EnsembleAppTarget
from targets.target_support import EnsembleTiming


def is_stored(value: SSAValue) -> bool:
    """True if value comes from a save_stationary."""
    return type(value) is not BlockArgument and type(value.owner) is cd.CimSaveStationaryOp


def rram_macros_per_copy(target: EnsembleAppTarget, shape: list[int], bits: int) -> int:
    """RRAM macros one copy of a stored value takes, as xdsl_hardware lays it out
    (_stationary_layout: a grouped [g, v, k] weight block-diagonally when a group fits a
    macro, else row tiles x column tiles of every matrix [in, out])."""
    unit = target.rram.array
    cells = -(-bits // unit.bits_per_cell)
    if len(shape) == 3:
        g, v, k = shape
        per_macro = min(unit.rows // v, unit.cols // (k * cells))
        if per_macro > 0:
            return -(-g // per_macro)
    if len(shape) == 1:
        blocks, in_dim, out_dim = 1, shape[0], 1
    else:
        blocks, in_dim, out_dim = math.prod(shape[:-2]), shape[-2], shape[-1]
    return blocks * -(-in_dim // unit.rows) * -(-out_dim * cells // unit.cols)


@dataclass(frozen=True)
class CorrectionPlan:
    """The SRAM correction of one stored matvec at one RRAM pf (pf RRAM copies, each of
    rram_per_copy macros).

      weights           correction weights: sram_share_pct of the matrix's weights
      originals         SRAM macros that hold them (dense: pe_count x weights_per_pe each)
      macros            SRAM macros the correction runs on, stationary beside the RRAM
                        copies: per copy, as many as keep pace with the RRAM MAC (each PE
                        adds MAC-cycles / cycles_per_weight weights per input bit), at most
                        sram_macros_per_rram_macro per RRAM macro; never fewer than the
                        originals
      cycles_per_bit    cycles the correction of one input row takes per input bit, on
                        macros // pf macros (the op's round takes the slower side)
    """
    weights: int
    originals: int
    macros: int
    cycles_per_bit: int


def correction_plan(target: EnsembleAppTarget, shape: list[int], bits: int, pf: int) -> CorrectionPlan:
    """CorrectionPlan of a stored matvec whose weight has this shape and bits, at RRAM pf."""
    s, r = target.sram, target.rram
    if s.cycle_ns != r.cycle_ns:
        raise ValueError(f"the SRAM ({s.cycle_ns} ns) and RRAM ({r.cycle_ns} ns) macros share one clock")
    weights = math.ceil(math.prod(shape) * target.sram_share_pct / 100)
    originals = -(-weights // (s.pe_count * s.weights_per_pe))
    per_pe = r.cycles_per_mac // s.cycles_per_weight
    if per_pe == 0:
        raise ValueError(f"an SRAM weight ({s.cycles_per_weight} cycles) is slower than one RRAM MAC "
                         f"({r.cycles_per_mac} cycles): the correction cannot keep pace")
    pace = -(-weights // (s.pe_count * per_pe))
    cap = math.floor(target.sram_macros_per_rram_macro * rram_macros_per_copy(target, shape, bits))
    macros = max(originals, pf * max(1, min(pace, cap)))
    per_copy = macros // pf
    cycles_per_bit = -(-weights // (s.pe_count * per_copy)) * s.cycles_per_weight
    return CorrectionPlan(weights=weights, originals=originals, macros=macros, cycles_per_bit=cycles_per_bit)


def dynamic_rhs_layout(target: EnsembleAppTarget, rhs_shape: list[int], rhs_bits: int) -> tuple[int, int]:
    """(SRAM macros one copy of a dynamic rhs [..., k, n] takes, rows per macro): k in row
    tiles of weights_per_pe, n over the pe_count PEs, each value in ceil(bits /
    weight_bits) slices, every block [k, n] on its own macros."""
    s = target.sram
    k, n = rhs_shape[-2], rhs_shape[-1]
    blocks = math.prod(rhs_shape[:-2])
    slices = -(-rhs_bits // s.weight_bits)
    macros = blocks * -(-k // s.weights_per_pe) * -(-n // s.pe_count) * slices
    return macros, min(k, s.weights_per_pe)


class EnsembleTimingMapping(TimingOnlyMapping):
    """Specs of a block on the RRAM/SRAM ensemble, given a pf per op (see the module
    docstring)."""

    def _init_timing(self, target: EnsembleAppTarget) -> None:
        if not isinstance(target, EnsembleAppTarget):
            raise TypeError("EnsembleTimingMapping needs the target bound to an app (ensemble_for_app)")
        self.target_timing = EnsembleTiming(target)

    def _plan_matvec(self, op: cd.CimMatvecOp):
        lhs, rhs = op.lhs, op.rhs
        lhs_shape = self._shape(lhs)
        if len(self._shape(rhs)) == 3:
            # grouped: one round per copy = all nb blocks at once
            row_dims = lhs_shape[:-2]
            row_size = lhs_shape[-2] * lhs_shape[-1]
        else:
            row_dims = lhs_shape[:-1]
            row_size = lhs_shape[-1]
        total_rows = math.prod(row_dims)
        # everything one round produces: for a tiled matvec ([..., tiles, out]) every row
        # tile's partial vector, since the tile macros run in parallel
        output_width = self._numel(op.result) // total_rows
        input_bits = bit_width(lhs)
        t, target = self.target_timing, self.target
        lhs_i, lhs_numel = self.node_index[lhs], self._numel(lhs)
        rhs_i, rhs_numel = self.node_index[rhs], self._numel(rhs)

        if is_stored(rhs):
            r = target.rram
            rhs_shape, rhs_bits = self._shape(rhs), bit_width(rhs)

            def recipe(pf):
                plan = correction_plan(target, rhs_shape, rhs_bits, pf)
                # per input bit, the slower of the RRAM MAC and its SRAM correction
                per_bit = max(r.cycles_per_mac, plan.cycles_per_bit) * r.cycle_ns
                return matvec_recipe(
                    lhs_i=lhs_i, lhs_numel=lhs_numel, rhs_i=rhs_i, rhs_numel=rhs_numel,
                    row_size=row_size, total_rows=total_rows, output_width=output_width,
                    hardware_interval=t.wmu_read() + input_bits * per_bit, pf=pf, setup=0.0,
                )
            return recipe

        s = target.sram
        _macros, rows = dynamic_rhs_layout(target, self._shape(rhs), bit_width(rhs))
        hardware_interval = t.wmu_read() + input_bits * rows * s.cycles_per_weight * s.cycle_ns
        # one macro's rows x pe_count columns x weight_bits, written write_word_bits per cycle
        write_ns = -(-rows * s.pe_count * s.weight_bits // s.write_word_bits) * s.cycle_ns

        def recipe_dynamic(pf):
            return matvec_recipe(
                lhs_i=lhs_i, lhs_numel=lhs_numel, rhs_i=rhs_i, rhs_numel=rhs_numel,
                row_size=row_size, total_rows=total_rows, output_width=output_width,
                hardware_interval=hardware_interval, pf=pf, setup=write_ns,
            )
        return recipe_dynamic

    # only the matvec is costed here; the base's other planners use single-module costs
    _SPEC_PLANNERS = {
        cd.CimMatvecOp: _plan_matvec,
        cd.CimReduceVecOp: TimingOnlyMapping._plan_reduce_vec,
        cd.CimVecElMulOp: TimingOnlyMapping._plan_vec_el_mul,
        cd.CimVecAddOp: TimingOnlyMapping._plan_vec_add,
        cd.CimLutSearchOp: TimingOnlyMapping._plan_lut_search,
        cd.CimLutSearchVectorOp: TimingOnlyMapping._plan_lut_search,
        cd.CimSoftmaxOp: TimingOnlyMapping._plan_row_op,
        cd.CimLayerNormOp: TimingOnlyMapping._plan_row_op,
        cd.CimRMSNormOp: TimingOnlyMapping._plan_row_op,
        cd.CimReluOp: TimingOnlyMapping._plan_pointwise,
        cd.CimRsqrtOp: TimingOnlyMapping._plan_pointwise,
        cd.CimGeluOp: TimingOnlyMapping._plan_pointwise,
        cd.CimSiluOp: TimingOnlyMapping._plan_pointwise,
        cd.CimEluOp: TimingOnlyMapping._plan_pointwise,
        cd.CimLeakyReluOp: TimingOnlyMapping._plan_pointwise,
        cd.CimQuantizeOp: TimingOnlyMapping._plan_pointwise,
        cd.CimDequantizeOp: TimingOnlyMapping._plan_pointwise,
        cd.CimArgmaxOp: TimingOnlyMapping._plan_topk,
        cd.CimMaxOp: TimingOnlyMapping._plan_topk,
        cd.CimScatterOp: TimingOnlyMapping._plan_scatter,
        cd.CimVecSelectOp: TimingOnlyMapping._plan_vec_select,
        cd.CimRowAbsmaxOp: TimingOnlyMapping._plan_row_absmax,
        cd.CimQuantizePerTokenOp: TimingOnlyMapping._plan_quantize_per_token,
    }
