"""Macro allocation of the RRAM/SRAM ensemble: EnsembleHardwareMapping.

Two pools of macros, sized by need (the paper sizes the SRAM per network, Fig. 11, 12):

  RRAM   the stored values (weights, LUT tables), one weight per cell (3-bit weights,
         kernels/w3.py), as HardwareMapping's RMU: each consumer's copies (one per RRAM
         pf) bin-packed by columns; total = max(stationary, peak of the ops' copies)
  SRAM   stationary  each weight matrix's correction macros at its op's pf
                     (correction_plan.macros: beside its RRAM copies, at most
                     sram_macros_per_rram_macro per RRAM macro), held for the whole run
         volatile    a dynamic matvec's macros, pf copies of its rhs (dynamic_rhs_layout),
                     from the start of its setup (writing them) to its end; and its
                     rhs's staging copy (one copy), written by the rhs's producer as it
                     comes out, from the rhs's first batch to the matvec's start (the
                     setup writes the copies from it; OperationMapping.rhs_hold_*);
                     peak over time
         total = stationary + peak volatile

The cap is on RRAM + SRAM macros (EnsembleArrayCounts.total_array_count). Other
activations between ops are not counted, as on the other targets.
"""
from __future__ import annotations

from dataclasses import dataclass

from xdsl.ir import Operation, SSAValue

from compiler.dialects import cim_dialect as cd
from compiler.dialects.rewrite_helpers import shape_of
from compiler.xdsl_ensemble_timing import correction_plan, dynamic_rhs_layout, is_stored, rram_macros_per_copy
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_hardware import HardwareArrayCounts, HardwareMapping, OperationMapping, VolatileDemand
from targets.target_loader import EnsembleAppTarget


@dataclass(frozen=True)
class EnsembleOperationMapping(OperationMapping):
    """OperationMapping with its macros per pool: volatile_arrays = rram + sram, while it
    runs; sram_correction_macros, a stored matvec's stationary SRAM correction."""
    rram_macros: int
    sram_macros: int
    sram_correction_macros: int


@dataclass(frozen=True)
class EnsembleArrayCounts:
    """Macro totals: each pool's counts, total = their sum."""
    total_array_count: int
    rram: HardwareArrayCounts
    sram: HardwareArrayCounts


class EnsembleHardwareMapping(HardwareMapping):
    """Macro placement of one already-timed block on the ensemble (see the module
    docstring)."""

    def _init_units(self, target: EnsembleAppTarget) -> None:
        if not isinstance(target, EnsembleAppTarget):
            raise TypeError("EnsembleHardwareMapping needs the target bound to an app (ensemble_for_app)")
        self.ensemble = target
        # stored values live in the RRAM macros (HardwareMapping's RMU); the SRAM macros
        # are counted here, not through the WMU
        self.rmu = target.rram.array
        self.wmu = None

    def map_operation(self, op: Operation) -> EnsembleOperationMapping:
        """The op's macros from the start of its setup (a dynamic matvec's: writing its
        rhs copies) to its end, and a dynamic matvec's staged rhs."""
        value = op.results[0]
        idx = self._mapper.node_index[value]
        pf = self._operation_pf(value)
        rram, sram, correction = self.macros_at_pf(op, pf)
        start = self._required_time(value, "start")
        hold = {}
        if idx in self._held_rhs:
            hold = {"rhs_hold_arrays": self._held_rhs_arrays(op.rhs),
                    "rhs_hold_start": self._required_time(op.rhs, "start"),
                    "rhs_hold_end": self._required_time(value, self.HELD_RHS_UNTIL)}
        return EnsembleOperationMapping(
            pf=pf, startup=start - self._mapper.setup_at(idx), output_end_time=self._required_time(value, "end"),
            volatile_arrays=rram + sram, rram_macros=rram, sram_macros=sram, sram_correction_macros=correction,
            **hold)

    # a dynamic matvec's rhs is staged in SRAM macros until it starts (HELD_RHS_UNTIL
    # "start", HardwareMapping's): its setup writes the pf copies from the staging copy
    def _holds_rhs(self, op: Operation) -> bool:
        return type(op) is cd.CimMatvecOp and not is_stored(op.rhs)

    def _held_rhs_arrays(self, rhs: SSAValue) -> int:
        """One copy of the rhs in SRAM macros (dynamic_rhs_layout)."""
        macros, _rows = dynamic_rhs_layout(self.ensemble, shape_of(rhs.type), bit_width(rhs))
        return macros

    # ---- pools of volatile macros, for array ids (xdsl_materialize)
    def volatile_pool(self, value: SSAValue) -> str:
        """A dynamic matvec's macros are SRAM; a stored matvec's or LUT's, RRAM."""
        return "sram" if self._holds_rhs(value.owner) else "rram"

    def volatile_pools(self, counts: EnsembleArrayCounts) -> list[tuple[str, int, int]]:
        return [(pool, getattr(counts, pool).volatile_array_count, getattr(counts, pool).total_array_count)
                for pool in ("rram", "sram")]

    def macros_at_pf(self, op: Operation, pf: int) -> tuple[int, int, int]:
        """(RRAM macros while it runs, SRAM macros while it runs, SRAM correction macros
        held for the whole run) of a mapped op at this pf: a stored matvec's RRAM copies
        and its correction, a dynamic matvec's SRAM copies of the rhs, a LUT's RRAM copies."""
        if type(op) is cd.CimMatvecOp and is_stored(op.rhs):
            rram = self._matvec_stationary_second_arrays(op, op.rhs, op.lhs, pf)
            shape, bits = shape_of(op.rhs.type), bit_width(op.rhs)
            if rram_macros_per_copy(self.ensemble, shape, bits) != self.stationary_arrays_per_copy(op.rhs):
                raise RuntimeError(f"{op.name}: rram_macros_per_copy and the RRAM layout disagree")
            return rram, 0, correction_plan(self.ensemble, shape, bits, pf).macros
        if type(op) is cd.CimMatvecOp:
            return 0, self._matvec_both_dynamic_arrays(op, op.lhs, op.rhs, pf), 0
        return self.volatile_arrays_at_pf(op, pf), 0, 0

    def _matvec_both_dynamic_arrays(self, op: cd.CimMatvecOp, lhs: SSAValue, rhs_value: SSAValue, pf: int) -> int:
        """pf copies of the dynamic rhs in SRAM macros (dynamic_rhs_layout)."""
        out_dim = self._matvec_out_dim(op)
        in_dim, rhs_out_dim = self._matvec_dynamic_rhs_dims(rhs_value)
        if out_dim != rhs_out_dim:
            raise RuntimeError("hardware matvec mapping: output last dim does not match rhs out dim")
        self._require_input_last_dim(lhs, in_dim)
        macros, _rows = dynamic_rhs_layout(self.ensemble, shape_of(rhs_value.type), bit_width(rhs_value))
        return pf * macros

    def array_counts(self, stationary_placements, operation_mappings: dict[SSAValue, OperationMapping]
                     ) -> EnsembleArrayCounts:
        """RRAM: max(its stationary copies bin-packed, peak of its ops' copies). SRAM: the
        correction macros plus the peak of its ops' volatile macros (the staged rhs among
        them). Total = both."""
        demands = {"rram": [], "sram": []}
        for m in operation_mappings.values():
            for pool, count in (("rram", m.rram_macros), ("sram", m.sram_macros)):
                if count > 0:
                    demands[pool].append(VolatileDemand(array_count=count, live_start=m.startup,
                                                        live_end=m.output_end_time))
            if m.rhs_hold_arrays > 0:
                demands["sram"].append(VolatileDemand(array_count=m.rhs_hold_arrays, live_start=m.rhs_hold_start,
                                                      live_end=m.rhs_hold_end))
        rram_stationary = self._bin_pack_stationary_arrays(list(stationary_placements), self.rmu.cols)
        rram_volatile = self._peak_volatile_array_count(demands["rram"])
        rram = HardwareArrayCounts(total_array_count=max(rram_stationary, rram_volatile),
                                   nonvolatile_array_count=rram_stationary, volatile_array_count=rram_volatile)
        sram_stationary = sum(m.sram_correction_macros for m in operation_mappings.values())
        sram_volatile = self._peak_volatile_array_count(demands["sram"])
        sram = HardwareArrayCounts(total_array_count=sram_stationary + sram_volatile,
                                   nonvolatile_array_count=sram_stationary, volatile_array_count=sram_volatile)
        return EnsembleArrayCounts(total_array_count=rram.total_array_count + sram.total_array_count,
                                   rram=rram, sram=sram)
