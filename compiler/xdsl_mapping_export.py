"""Exports a mapped block (after the search) as a JSON-ready dict.

Holds everything a later pass needs to compute tiles, data movement between
tiles, buffers, area and energy without compiling or searching again:

  args   every block argument: shape and bits per element.
  ops    every op: its attributes, output shape and bits, pf, schedule (when
         its output batches come out), its input edges, and what it occupies:
           units       {sub-kind: circuits} it holds (accelerator ops;
                       accelerator_demands, a softmax op one per stage)
           volatile    WMU arrays and their live range (mapped ops; from the start of
                       the op's setup); a matvec with a staged dynamic rhs (1F1C) also
                       the staging copy, rhs_hold_arrays from rhs_hold_start_ns to
                       rhs_hold_end_ns (xdsl_hardware.OperationMapping)
           stationary  per resident operand: copies, tile layout, live range
  accelerators  circuits provisioned per sub-kind for the whole chip
                (compute_accel_peak_concurrency: shared over time, reduce trees
                borrowing across sizes).

Ids are the mapper's node positions; an input edge's "from" is the producer's id.
A schedule's unit_size is null when the whole output comes out at once.
"""
import math

from xdsl.dialects.builtin import IntegerAttr, StringAttr
from xdsl.ir import SSAValue

from compiler.dialects.rewrite_helpers import shape_of
from compiler.xdsl_accel_concurrency import ACCEL_OP_KINDS, accelerator_demands, compute_accel_peak_concurrency
from compiler.xdsl_dual_module_hardware import DualModuleArrayCounts
from compiler.xdsl_ensemble_hardware import EnsembleArrayCounts, EnsembleOperationMapping
from compiler.xdsl_fixed_point import bit_width
from compiler.xdsl_hardware import HardwareArrayCounts
from compiler.xdsl_search import SearchResult
from compiler.xdsl_timing import _is_return


def _numel(value: SSAValue) -> int:
    n = 1
    for d in shape_of(value.type):
        n *= d
    return n


def _attr_value(attr):
    if isinstance(attr, IntegerAttr):
        return attr.value.data
    if isinstance(attr, StringAttr):
        return attr.data
    return str(attr)


def _array_counts(counts: HardwareArrayCounts | DualModuleArrayCounts | EnsembleArrayCounts) -> dict:
    match counts:
        case HardwareArrayCounts():
            return {"total": counts.total_array_count, "nonvolatile": counts.nonvolatile_array_count,
                    "volatile": counts.volatile_array_count}
        case DualModuleArrayCounts():
            return {"total": counts.total_array_count, "analog": _array_counts(counts.analog),
                    "digital": _array_counts(counts.digital)}
        case EnsembleArrayCounts():
            return {"total": counts.total_array_count, "rram": _array_counts(counts.rram),
                    "sram": _array_counts(counts.sram)}
        case _:
            raise TypeError(f"unknown array counts type {type(counts).__name__}")


def export_mapping(block, result: SearchResult) -> dict:
    mapper, hardware, array_counts = result.mapper, result.hardware, result.array_counts
    node_index = mapper.node_index
    specs = mapper.specs
    operation_mappings = hardware.rebuild_operations()

    stationary_of: dict = {}
    for stat_value, consumer_op, placement in hardware.stationary_placements_by_pair():
        stationary_of.setdefault(consumer_op, []).append({
            "value": node_index[stat_value],
            "copies": placement.copies,
            "full_arrays_per_copy": placement.full_arrays_per_copy,
            "partial_tile_cols": placement.partial_tile_cols,
            "partial_tiles_per_copy": placement.partial_tiles_per_copy,
            "live_start_ns": hardware.stationary_live_start(stat_value),
            "live_end_ns": hardware.stationary_live_end(consumer_op),
        })

    args = [{"id": node_index[a], "shape": list(shape_of(a.type)), "type": str(a.type.element_type),
             "bits": bit_width(a)} for a in block.args]

    ops = []
    for op in block.ops:
        if _is_return(op):
            continue
        value = op.results[0]
        idx = node_index[value]
        spec = specs[idx]
        pf = mapper.mapped_pf[idx]
        entry = {
            "id": idx,
            "op": op.name,
            "attributes": {name: _attr_value(a) for name, a in {**op.properties, **op.attributes}.items()},
            "shape": list(shape_of(value.type)),
            "type": str(value.type.element_type),
            "bits": bit_width(value),
            "pf": pf,
            "schedule": {"start_ns": spec.start, "end_ns": spec.end, "interval_ns": spec.interval,
                         "unit_size": None if math.isinf(spec.unit_size) else spec.unit_size,
                         "count": spec.count},
            "inputs": [{"from": node_index[o], "elements": _numel(o), "bits": bit_width(o)} for o in op.operands],
        }
        if type(op) in ACCEL_OP_KINDS:
            entry["units"] = accelerator_demands(op, pf)
        if value in operation_mappings:
            m = operation_mappings[value]
            entry["volatile"] = {"arrays": m.volatile_arrays, "live_start_ns": m.startup,
                                 "live_end_ns": m.output_end_time}
            if m.rhs_hold_arrays > 0:
                # the dynamic rhs's staging copy, besides "arrays"
                entry["volatile"]["rhs_hold_arrays"] = m.rhs_hold_arrays
                entry["volatile"]["rhs_hold_start_ns"] = m.rhs_hold_start
                entry["volatile"]["rhs_hold_end_ns"] = m.rhs_hold_end
            if isinstance(m, EnsembleOperationMapping):
                entry["volatile"]["rram_macros"] = m.rram_macros
                entry["volatile"]["sram_macros"] = m.sram_macros
                entry["sram_correction_macros"] = m.sram_correction_macros
        if op in stationary_of:
            entry["stationary"] = stationary_of[op]
        ops.append(entry)

    return {
        "latency_ns": result.latency_ns,
        "arrays": _array_counts(array_counts),
        "accelerators": compute_accel_peak_concurrency(block, specs, mapper.mapped_pf, node_index),
        "args": args,
        "ops": ops,
    }
