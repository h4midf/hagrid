"""Timing model of a dual-module target (HyFlexPIM): DualModuleTimingMapping.

The spec propagation, pf caps and memos are TimingOnlyMapping's. Only the op costs
differ: each op is costed on its module, fixed by its operands (op_module):

  a stored value (weight, LUT table)   lives in the analog module; every matvec or
                                       lookup that uses it runs there
  a matvec, both operands dynamic      runs in the digital module

as in the paper's mapping (HyFlexPIM Fig. 9). LUT tables in the analog module are our
addition (the paper runs non-linear ops on the SFU).

Op costs modeled so far:
  matvec, stored rhs (analog)   one input row per round: input bits sent one per
                                100 ns cycle (HyFlexPIM Sec. 5.4, design.json "input");
                                the ADC converts the held values while the crossbar
                                reads the next bit, so a round is input_bits x 100 ns.
                                Every row and column tile of the matrix runs in its own
                                array at once; the row tiles' partial sums are added by
                                the reduce_vec the tiling puts after the matvec.
  matvec, both dynamic          pf counts compute arrays. Up to one per output: outputs
  (digital)                     take turns on them, one per round, so one input row's
                                outputs spread over many arrays (as the paper's module
                                throughput, 256 arrays x 1024 / (64 x 3) / 5 outputs per
                                cycle, spreads them). Beyond that, each output spreads over
                                g = pf // outputs arrays (at most K): each computes and adds
                                ceil(K / g) products, then the partial sums are added in a
                                tree of ceil(log2 g) levels, one addition row each. Each
                                output is a dot product of K terms (K = the input row's length):
                                K int8 x int8 multiplies and K - 1 additions, all in NOR
                                logic, each NOR taking 3 columns of a row (HyFlexPIM Sec.
                                3.1). A multiply is 64 NORs (192 columns, so a 1024-column
                                row holds 5); an addition of the accumulator's width,
                                16 + ceil(log2 K) bits, is 13 NORs per bit (FloatPIM Sec.
                                6.1). A row costs its 4 write and 1 read cycles of 1.1 ns
                                (Sec. 3.1: the NORs themselves add no cycles there). On one
                                array, a round is ceil(K / multiplies per row) +
                                ceil((K - 1) / additions per row) rows.
  reduce_vec, vec_el_mul,       as on a single-module target (TimingOnlyMapping): the
  vec_add                       same custom_op_library reduce tree / vec_add / vec_mul
                                units, at
                                the same technology. The WMU read / write around them is
                                the digital module's (the paper accumulates partial sums
                                there, Sec. 3.3): a row read (1 cycle) and a row write
                                (4 cycles) of 1.1 ns (Sec. 3.1, FloatPIM).
  softmax, layer_norm,          as on a single-module target: one row per round through
  rms_norm                      the custom_op_library row circuit, between a WMU read
                                and write (the digital module's row read / write). This
                                replaces the paper's SFU with the same units as 1F1C.
  relu, gelu, silu, elu,        as on a single-module target: one element per round
  leaky_relu, quantize,         through the custom_op_library circuit, between a WMU
  dequantize                    read and write (the digital module's).
  argmax, max, scatter          as on a single-module target: argmax / max are k
                                comparator-tree passes between a WMU read and write;
                                scatter is a WMU write.
  row_absmax, quantize_per_     as on a single-module target: the token_quant abs-max
  token, vec_select             tree, token_scale + quant_mant_round, and the vec_select
                                multiplexers, between WMU reads and writes.
  lut_search(_vector)           as on a single-module target: an RMU read of the table
                                entry (one analog row read out, one 100 ns cycle) plus a
                                WMU read of the index (a digital row read).
The array operations come from DualModuleTiming (targets/target_support.py).
Every other op with a cost raises (TimingOnlyMapping._build_spec_uncached).
"""
from __future__ import annotations

import math

from xdsl.ir import BlockArgument, Operation, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_timing import PipelineRecipe, TimingOnlyMapping, matvec_recipe
from targets.target_loader import DualModuleTarget
from targets.target_support import DualModuleTiming

MODULES = ("analog", "digital")


def is_stored(value: SSAValue) -> bool:
    """True if value comes from a save_stationary."""
    return type(value) is not BlockArgument and type(value.owner) is cd.CimSaveStationaryOp


def op_module(op: Operation) -> str:
    """Module a matvec or LUT lookup runs on: analog if it uses a stored value (a
    matvec's rhs, a LUT table), digital for a matvec with both operands dynamic."""
    if type(op) is cd.CimMatvecOp:
        return "analog" if is_stored(op.rhs) else "digital"
    if type(op) in (cd.CimLutSearchOp, cd.CimLutSearchVectorOp):
        return "analog"
    raise ValueError(f"op_module: {op.name} is not a matvec or LUT lookup")


class DualModuleTimingMapping(TimingOnlyMapping):
    """Specs of a block on a dual-module target, given a pf per op (see the module
    docstring)."""

    def _init_timing(self, target: DualModuleTarget) -> None:
        self.analog = target.analog
        self.digital = target.digital
        # the same custom_op_library accelerator units as a single-module target, and
        # this target's array reads / writes under the RMU / WMU names
        self.target_timing = DualModuleTiming(target)

    def _digital_rows_per_output(self, op: cd.CimMatvecOp, k: int, output_width: int, arrays: int) -> int:
        """Digital-array rows of one output on `arrays` compute arrays: each takes
        ceil(k / arrays) of its k products, multiplies and adds them, then the partial sums
        are added in a tree of ceil(log2 arrays) levels of one addition each. Each NOR
        takes logic_op_columns of a row (see the module docstring)."""
        d = self.digital
        lhs_bits, rhs_bits = bit_width(op.lhs), bit_width(op.rhs)
        if (lhs_bits, rhs_bits) != (8, 8):
            raise NotImplementedError(f"digital matvec of {lhs_bits} x {rhs_bits} bits: only int8 x int8 is costed")
        if output_width != self._shape(op.rhs)[-1]:
            raise NotImplementedError("digital matvec: a tiled dynamic rhs is not costed")
        acc_bits = lhs_bits + rhs_bits + math.ceil(math.log2(k))
        mul_per_row = d.array.cols // (d.int8_mul_logic_ops * d.logic_op_columns)
        add_per_row = d.array.cols // (d.add_logic_ops_per_bit * acc_bits * d.logic_op_columns)
        if add_per_row == 0:
            raise ValueError(f"a {acc_bits}-bit addition needs more than one {d.array.cols}-column row")
        share = -(-k // arrays)
        levels = math.ceil(math.log2(arrays)) if arrays > 1 else 0
        return -(-share // mul_per_row) + -(-(share - 1) // add_per_row) + levels

    def _pf_cap_for_value_uncached(self, value: SSAValue) -> int:
        """As TimingOnlyMapping, except a digital matvec: its pf counts compute arrays, up
        to one per product (outputs x K), since an output's products and additions can
        spread over several arrays."""
        op = value.owner if type(value) is not BlockArgument else None
        if type(op) is cd.CimMatvecOp and op_module(op) == "digital":
            return self._numel(op.result) * self._shape(op.lhs)[-1]
        return super()._pf_cap_for_value_uncached(value)

    def _plan_matvec(self, op: cd.CimMatvecOp):
        """Analog: one input row per round. Digital: one output per round, so the outputs
        of one input row spread over pf compute arrays (see the module docstring)."""
        lhs, rhs = op.lhs, op.rhs
        lhs_shape = self._shape(lhs)
        if len(self._shape(rhs)) == 3:
            # grouped: one round per copy = all nb blocks at once
            row_dims = lhs_shape[:-2]
            row_size = lhs_shape[-2] * lhs_shape[-1]
        else:
            row_dims = lhs_shape[:-1]
            row_size = lhs_shape[-1]
        total_rows = 1
        for d in row_dims:
            total_rows *= d
        # everything one round produces: for a tiled matvec ([..., tiles, out]) every row
        # tile's partial vector, since the tile arrays run in parallel
        output_width = self._numel(op.result) // total_rows
        lhs_i, lhs_numel = self.node_index[lhs], self._numel(lhs)
        rhs_i, rhs_numel = self.node_index[rhs], self._numel(rhs)

        if op_module(op) == "digital":
            outputs = total_rows * output_width
            row_ns = self.target_timing.digital_row_ns()

            def recipe_digital(pf):
                # up to one array per output, outputs taking turns; beyond that, each
                # output spreads over pf // outputs arrays (at most one per product)
                arrays = min(row_size, pf // outputs) if pf > outputs else 1
                per_output = self._digital_rows_per_output(op, row_size, output_width, arrays) * row_ns
                return PipelineRecipe(
                    first_needs=((rhs_i, rhs_numel), (lhs_i, row_size)),
                    all_needs=((rhs_i, rhs_numel), (lhs_i, lhs_numel)),
                    num_problems=outputs,
                    batch_terms=((lhs_i, 1, row_size, output_width),),
                    outputs_per_problem=1, hardware_interval=per_output, pf=min(pf, outputs), setup=0.0,
                )
            return recipe_digital

        hardware_interval = bit_width(lhs) * self.analog.cycle_ns

        def recipe(pf):
            return matvec_recipe(
                lhs_i=lhs_i, lhs_numel=lhs_numel,
                rhs_i=rhs_i, rhs_numel=rhs_numel,
                row_size=row_size, total_rows=total_rows,
                output_width=output_width,
                hardware_interval=hardware_interval, pf=pf, setup=0.0,
            )
        return recipe

    # only the ops costed above; the base's other planners use single-module costs
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
