"""Array placement of a dual-module target (HyFlexPIM): DualModuleHardwareMapping.

The chip has two modules with their own arrays: the analog module (64 x 128 arrays,
SLC / MLC) and the digital module (1024 x 1024 SLC arrays). The module of an op is
fixed by its operands (compiler/xdsl_dual_module_timing.py, op_module):

  a stored value (weight, LUT table)   lives in the analog module; every matvec or
                                       lookup that uses it runs there
  a matvec, both operands dynamic      runs in the digital module

as in the paper's mapping (HyFlexPIM Fig. 9): weights in the analog module, Q x K^T and
x V in the digital module. LUT tables in the analog module are our addition, not the
paper's (the paper runs non-linear ops on the SFU): a lookup there is a one-hot matvec,
the index drives one row and the bitlines read that row out.

The cim block is tiled at lowering to each matvec's module rows (compiler/lowering.py);
a stored value's arrays come from its shape here (HardwareMapping._tile_layout).

Each module is its own pool, counted as in HardwareMapping (max of the bin-packed
stationary arrays and the peak volatile arrays); the total is the sum of both modules.

A matvec's weights are split by output column: target.analog.slc_rate_pct of them in SLC,
the rest in MLC (_stationary_layout). LUT tables stay in SLC.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.rewrite_helpers import shape_of
from compiler.xdsl_dual_module_timing import MODULES, op_module
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_hardware import HardwareArrayCounts, HardwareMapping, OperationMapping, VolatileDemand
from targets.target_loader import DualModuleTarget


@dataclass(frozen=True)
class DualModuleArrayCounts:
    """Array totals of a dual-module mapping: each module's own counts, total = their sum."""
    total_array_count: int
    analog: HardwareArrayCounts
    digital: HardwareArrayCounts


class DualModuleHardwareMapping(HardwareMapping):
    """Array placement of one already-timed block on a dual-module target (see the
    module docstring)."""

    def _init_units(self, target: DualModuleTarget) -> None:
        self.units = {"analog": target.analog.array, "digital": target.digital.array}
        self.analog_module = target.analog
        # the single-module RMU / WMU do not exist here
        self.rmu = self.wmu = None

    # ---- placement
    def _stationary_layout(self, stat_value: SSAValue) -> tuple[int, int, int]:
        """Tiles of one resident copy in the analog arrays, as HardwareMapping._tile_layout:
        in_dim on the rows, the columns side by side. A matvec's weights are split by
        output column: slc_rate_pct of them in SLC (slc_bits per cell), the rest in MLC
        (mlc_bits per cell), in the same arrays (the ADC switches mode per conversion,
        HyFlexPIM Sec. 3.2). A LUT table stays in SLC."""
        shape = tuple(shape_of(stat_value.type))
        bits = bit_width(stat_value)
        weights = all(type(use.operation) is cd.CimMatvecOp for use in stat_value.uses)
        key = (shape, bits, weights)
        cached = self._stationary_layout_cache.get(key)
        if cached is not None:
            return cached
        if len(shape) == 1:
            blocks, in_dim, out_dim = 1, shape[0], 1
        else:
            blocks = 1
            for d in shape[:-2]:
                blocks *= d
            in_dim, out_dim = shape[-2], shape[-1]
        unit, am = self.units["analog"], self.analog_module
        slc_cols = -(-out_dim * am.slc_rate_pct // 100) if weights else out_dim
        cols = slc_cols * -(-bits // am.slc_bits) + (out_dim - slc_cols) * -(-bits // am.mlc_bits)
        full_col_tiles, partial_cols = divmod(cols, unit.cols)
        strips = blocks * -(-in_dim // unit.rows)
        result = (strips * full_col_tiles, partial_cols, strips if partial_cols else 0)
        self._stationary_layout_cache[key] = result
        return result

    def _matvec_both_dynamic_arrays(self, op: cd.CimMatvecOp, lhs: SSAValue, rhs_value: SSAValue, pf: int) -> int:
        """Matvec with both operands dynamic on the digital module (HyFlexPIM Sec. 3.1,
        3.3): pf compute arrays, one per parallel copy, while it runs. Its rows compute by
        in-row NOR (FloatPIM): an output takes 64 NORs of 3 columns each, 192 columns, so
        a row holds about 5 outputs. The operands are written into the row being
        processed, so a compute array holds no matrix. The rhs is stored once besides,
        held from its first batch (_held_rhs_arrays)."""
        out_dim = self._matvec_out_dim(op)
        in_dim, rhs_out_dim = self._matvec_dynamic_rhs_dims(rhs_value)
        if out_dim != rhs_out_dim:
            raise RuntimeError("hardware matvec mapping: output last dim does not match rhs out dim")
        self._require_input_last_dim(lhs, in_dim)
        return pf

    def volatile_pool(self, value: SSAValue) -> str:
        """A mapped op's volatile arrays are in its module."""
        return op_module(value.owner)

    def volatile_pools(self, counts: DualModuleArrayCounts) -> list[tuple[str, int, int]]:
        """Each module, analog first: (module, its peak volatile arrays, its arrays)."""
        return [(m, getattr(counts, m).volatile_array_count, getattr(counts, m).total_array_count) for m in MODULES]

    # the rhs's only copy: written by its producer as it comes out, read by the compute
    # arrays until the matvec ends
    HELD_RHS_UNTIL = "end"

    def _holds_rhs(self, op) -> bool:
        """Every matvec with both operands dynamic (the digital module's) holds its rhs."""
        return type(op) is cd.CimMatvecOp and not self._is_stationary_value(op.rhs)

    def _held_rhs_arrays(self, rhs: SSAValue) -> int:
        """The dynamic rhs (K^T, V) stored once in digital SLC arrays, packed by bits:
        ceil(elements * bits / cells per array)."""
        unit = self.units["digital"]
        elements = 1
        for d in shape_of(rhs.type):
            elements *= d
        cells = elements * self._cells_per_element(bit_width(rhs), unit)
        return -(-cells // (unit.rows * unit.cols))

    # ---- totals
    def array_counts(self, stationary_placements, operation_mappings: dict[SSAValue, OperationMapping]
                     ) -> DualModuleArrayCounts:
        """Analog: the stationary copies bin-packed into its arrays, and the peak of its
        ops' volatile demands. Digital: the peak of its ops' volatile demands, each stored
        dynamic rhs among them from its first batch. Total = analog + digital."""
        demands = {m: [] for m in MODULES}
        for value, m in operation_mappings.items():
            module = op_module(value.owner)
            if m.volatile_arrays > 0:
                demands[module].append(
                    VolatileDemand(array_count=m.volatile_arrays, live_start=m.startup, live_end=m.output_end_time))
            if m.rhs_hold_arrays > 0:
                # the stored rhs, in the matvec's module (digital), from its first batch
                demands[module].append(
                    VolatileDemand(array_count=m.rhs_hold_arrays, live_start=m.rhs_hold_start, live_end=m.rhs_hold_end))
        nonvolatile = {"analog": self._bin_pack_stationary_arrays(list(stationary_placements),
                                                                  self.units["analog"].cols),
                       "digital": 0}
        counts = {}
        for module in MODULES:
            volatile = self._peak_volatile_array_count(demands[module])
            counts[module] = HardwareArrayCounts(total_array_count=max(nonvolatile[module], volatile),
                                                 nonvolatile_array_count=nonvolatile[module],
                                                 volatile_array_count=volatile)
        return DualModuleArrayCounts(
            total_array_count=counts["analog"].total_array_count + counts["digital"].total_array_count,
            analog=counts["analog"], digital=counts["digital"])
