"""Energy of one inference of a mapped 1F1C model, per part.

Runs on a mapping (results/fefet1c_greedy/<model>/cap_<cap>.json) and its placement
(results/fefet1c_tiles/<model>/cap_<cap>.json); it never maps, searches or places.
Event counts follow each op's hardware round in compiler/xdsl_timing.py (_plan_*), so
energy and latency count the same events; per-event energies come from
TARGET.array_energy, accelerator units from xdsl_accel_concurrency.per_kind_accel_rows
of the model's block (rebuilt with the mapping's own flags).

Dynamic energy per op (every row / element it processes):
  matvec (every kind)         per row: the input row written into the PE input register
                              and read out bit-serially (256-bit accesses)
  matvec, weights in the RMU  per row: WMU read of the input row; per input bit, an RMU
                              MAC on every used column of a weight copy's tiles (column
                              charging + one ADC conversion), the arrays' phase control
                              (mac_cycles), shift-and-add (shift_add_cycles_per_mac) and,
                              with the weight-flip encoding, unit column (unit_column_mac_pJ)
  matvec, dynamic rhs         per row: WMU read of the input row; per input bit, a WMU
                              MAC (one ADC conversion per column) with phase control and
                              shift-and-add; once the row's MACs are done, every cell of
                              the rhs matrix read and written twice (the destructive WMU
                              MAC's rewrite)
  lut_search                  per lookup: RMU read (cell load) and WMU read of the row
  accelerator ops             WMU read of the dynamic inputs, one circuit evaluation per
                              element / reduction pass / softmax stage step, WMU write of
                              the output
  layout ops                  none (reinterpretations, as in the timing model)
A WMU read costs wmu_read_conversions ADC conversions per cell plus the WMU readout
logic; a WMU write wmu_write_cell_pJ per cell.
Data movement per access, from the placed traffic: router flit-hops and tile bus
transfers (ISAAC's full-activity power / its clock per access, TARGET.organization).
Dynamic energy only: leakage is not counted. Also not counted: TARGET's energy.json
"excluded".

Writes results/fefet1c_energy/<model>/cap_<cap>.json.

The target is TARGET_NAME (environment variable, required; targets.target_support reads
it at import time), and must be the mappings' target.

Usage (from the repo root):
  TARGET_NAME=1fefet1c/default python3 -m run_scripts.fefet1c_energy --model vit-large
"""
import argparse
import glob
import json
import math
import os

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import (
    _TOKEN_ROUND_SUB_KIND, _TOKEN_SCALE_SUB_KIND, _softmax_stage_sub_kind, per_kind_accel_rows,
)
from compiler.xdsl_timing import ABSMAX_TREE_INPUTS
from custom_op_library.common.tools import load_results
from run_scripts.fefet1c_tiles import RESULTS_DIR, TILES_DIR
from targets.target_support import TARGET

ENERGY_DIR = os.path.join(RESULTS_DIR, "fefet1c_energy")

_LAYOUT_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape",
               # no energy of their own: the timing model passes their operand on, and a
               # conv's im2col copies are read by its matvec (counted there, per element)
               "cim.transpose", "cim.unfold"}
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}
# accelerator ops evaluated once per output element
_ELEMENTWISE_OPS = {"cim.vec_add", "cim.vec_el_mul", "cim.vec_select", "cim.gelu", "cim.relu", "cim.silu",
                    "cim.elu", "cim.leaky_relu", "cim.quantize", "cim.dequantize", "cim.rsqrt"}


class _Energy:
    """Energy (pJ) by part of one mapping."""

    def __init__(self, unit_rows: dict):
        self.e = TARGET.array_energy
        self.unit_rows = unit_rows
        self.cells_per_row = TARGET.geometry.wmu.cols
        self.pJ: dict[str, float] = {}

    def add(self, part: str, pj: float):
        self.pJ[part] = self.pJ.get(part, 0.0) + pj

    def cells(self, elements: int, bits: int) -> int:
        return elements * -(-bits // TARGET.geometry.wmu.bits_per_cell)

    def wmu_read(self, cells: int, times: float = 1):
        e = self.e
        self.add("WMU read: ADC conversions", times * cells * e.wmu_read_conversions * e.adc_conversion_pJ)
        array_rows = -(-cells // self.cells_per_row)
        self.add("WMU read: readout logic",
                 times * array_rows * e.wmu_read_conversions * e.conversion_cycles * e.wmu_readout_cycle_pJ)

    def wmu_write(self, cells: int, times: float = 1):
        self.add("WMU write", times * cells * self.e.wmu_write_cell_pJ)

    def macs(self, arrays: int, times: float):
        """Phase control, shift-and-add and unit column of `arrays` arrays doing one MAC,
        `times` times."""
        e = self.e
        self.add("MAC: phase control", times * arrays * e.mac_cycles * e.macro_ctrl_cycle_pJ)
        self.add("MAC: shift-and-add", times * arrays * e.shift_add_cycles_per_mac * e.shift_add_cycle_pJ)
        self.add("MAC: unit column", times * arrays * e.unit_column_mac_pJ)

    def evaluations(self, sub_kind: str, count: int):
        if sub_kind not in self.unit_rows:
            raise KeyError(f"no row for accelerator unit {sub_kind!r}")
        pj = self.unit_rows[sub_kind]["dynamic_mW_est"] / self.e.cmos_energy_factor
        self.add(f"accelerator {sub_kind}", count * pj)


def _shape(mapping, ops, value_id) -> list[int]:
    if value_id in ops:
        return ops[value_id]["shape"]
    for a in mapping["args"]:
        if a["id"] == value_id:
            return a["shape"]
    raise KeyError(f"value {value_id} not found")


def _is_weight(ops, value_id) -> bool:
    return value_id in ops and ops[value_id]["op"] in _WEIGHT_OPS


def _matvec(en: _Energy, o, mapping, ops):
    lhs, rhs = o["inputs"]
    lhs_shape = _shape(mapping, ops, lhs["from"])
    rhs_shape = _shape(mapping, ops, rhs["from"])
    if len(rhs_shape) == 3:  # grouped: one round per copy covers all blocks
        rows, row_size = math.prod(lhs_shape[:-2]), lhs_shape[-2] * lhs_shape[-1]
    else:
        rows, row_size = math.prod(lhs_shape[:-1]), lhs_shape[-1]
    input_bits = lhs["bits"]
    en.wmu_read(en.cells(row_size, input_bits), rows)
    # each input row is written into its PE's input register and read out bit-serially
    ir_bits = TARGET.organization.pe_input_register_access_bits
    ir_accesses = rows * (-(-(row_size * input_bits) // ir_bits) + input_bits * -(-row_size // ir_bits))
    en.add("PE input registers", ir_accesses * TARGET.organization.pe_input_register_access_pJ)
    e = en.e
    if "stationary" in o:
        for s in o["stationary"]:
            columns = s["full_arrays_per_copy"] * TARGET.geometry.rmu.cols + s["partial_tiles_per_copy"] * s["partial_tile_cols"]
            arrays = s["full_arrays_per_copy"] + s["partial_tiles_per_copy"]
            macs = rows * input_bits
            en.add("RMU MAC: column charging", macs * columns * e.column_mac_pJ)
            en.add("RMU MAC: ADC conversions", macs * columns * e.adc_conversion_pJ)
            en.macs(arrays, macs)
        return
    if _is_weight(ops, rhs["from"]):
        raise RuntimeError(f"op {o['id']}: weight rhs without a stationary placement")
    # dynamic rhs [..., K, N]: per input bit a WMU MAC; per input row, once its MACs are
    # done, the K x N matrix rewritten (the destructive WMU MAC's restore)
    k, n = rhs_shape[-2], rhs_shape[-1]
    matrix_cells = en.cells(k * n, rhs["bits"])
    columns = en.cells(n, rhs["bits"]) * -(-k // TARGET.geometry.wmu.rows)
    arrays = -(-k // TARGET.geometry.wmu.rows) * -(-en.cells(n, rhs["bits"]) // TARGET.geometry.wmu.cols)
    macs = rows * input_bits
    en.wmu_read(matrix_cells, rows)
    en.wmu_write(matrix_cells, 2 * rows)
    en.add("WMU MAC: ADC conversions", macs * columns * e.wmu_mac_conversions * e.adc_conversion_pJ)
    en.macs(arrays, macs)


def _lut_search(en: _Energy, o, ops):
    index = [i for i in o["inputs"] if not _is_weight(ops, i["from"])]
    if len(index) != 1:
        raise RuntimeError(f"op {o['id']} (lut_search): expected one dynamic input, got {len(index)}")
    lookups = index[0]["elements"]
    row_cells = en.cells(math.prod(o["shape"]) // lookups, o["bits"])
    en.add("RMU read (LUT): cell load", lookups * row_cells * en.e.cell_load_pJ)
    en.wmu_read(row_cells, lookups)


def _accelerator(en: _Energy, o, ops):
    kind = o["op"]
    for inp in o["inputs"]:
        if not _is_weight(ops, inp["from"]):
            en.wmu_read(en.cells(inp["elements"], inp["bits"]))
    numel = math.prod(o["shape"])
    en.wmu_write(en.cells(numel, o["bits"]))
    if kind in _ELEMENTWISE_OPS:
        (sub_kind,) = o["units"]
        en.evaluations(sub_kind, numel)
    elif kind == "cim.reduce_vec":
        if o["attributes"]["reduce_impl"] != "tree":
            raise NotImplementedError(f"op {o['id']}: reduce_impl {o['attributes']['reduce_impl']!r}")
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        passes = -(-reduce_size // o["attributes"]["reduce_width"])
        en.evaluations(sub_kind, numel * passes)
    elif kind == "cim.max":
        # one argmax-tree evaluation per output element and pass of argmax_width inputs
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        passes = -(-reduce_size // o["attributes"]["argmax_width"])
        en.evaluations(sub_kind, numel * passes)
    elif kind == "cim.row_absmax":
        # one abs-max tree evaluation per row and pass of ABSMAX_TREE_INPUTS inputs
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        en.evaluations(sub_kind, numel * -(-reduce_size // ABSMAX_TREE_INPUTS))
    elif kind == "cim.quantize_per_token":
        # one token_scale per row, one rounding per element
        en.evaluations(_TOKEN_SCALE_SUB_KIND, numel // o["shape"][-1])
        en.evaluations(_TOKEN_ROUND_SUB_KIND, numel)
    elif kind == "cim.softmax":
        row_width = o["shape"][-1]
        rows = numel // row_width
        for _, module, row, per_element, passes in load_results.softmax_stages(o["attributes"]["variant"], row_width):
            sub_kind = _softmax_stage_sub_kind(module, row, o["inputs"][0]["bits"])
            en.evaluations(sub_kind, rows * row_width if per_element else rows * passes)
    elif kind == "cim.layer_norm":
        # the streamed circuit: every cycle of a row (dynamic_mW_est is pJ per cycle)
        (sub_kind,) = o["units"]
        rows = numel // o["shape"][-1]
        en.evaluations(sub_kind, rows * en.unit_rows[sub_kind]["cycles_per_row"])
    else:
        raise NotImplementedError(f"op {o['id']} ({kind}): no energy rule")


def energy_breakdown(mapping: dict, tiles: dict, unit_rows: dict) -> dict:
    """{part: pJ} of one inference."""
    en = _Energy(unit_rows)
    ops = {o["id"]: o for o in mapping["ops"]}
    for o in mapping["ops"]:
        kind = o["op"]
        if kind in _LAYOUT_OPS or kind in _WEIGHT_OPS:
            continue
        if kind == "cim.MatvecOp":
            _matvec(en, o, mapping, ops)
        elif kind == "cim.lut_search":
            _lut_search(en, o, ops)
        elif "units" in o:
            _accelerator(en, o, ops)
        else:
            raise NotImplementedError(f"op {o['id']} ({kind}): no energy rule")

    # data movement, per access, from the placed traffic (fefet1c_tiles.py)
    org = TARGET.organization
    traffic = tiles["traffic"]
    en.add("routers", traffic["flit_hops"] * org.router_flit_hop_pJ)
    en.add("tile buses", -(-traffic["bus_bits"] // org.tile_bus_bits) * org.tile_bus_transfer_pJ)
    return en.pJ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.model, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no tile results in {os.path.join(TILES_DIR, args.model)}")
    out_dir = os.path.join(ENERGY_DIR, args.model)
    os.makedirs(out_dir, exist_ok=True)

    rows_by_flags: dict[str, dict] = {}
    for path in paths:
        with open(path) as f:
            tiles = json.load(f)
        with open(tiles["mapping"]) as f:
            mapping = json.load(f)
        if tiles["target"] != TARGET.name or mapping["target"] != TARGET.name:
            raise ValueError(f"{path} is for {tiles['target']}, not {TARGET.name}")
        flags_key = json.dumps(mapping["flags"], sort_keys=True)
        if flags_key not in rows_by_flags:
            rows_by_flags[flags_key] = per_kind_accel_rows(build_tiled_cim_block(mapping["model"], mapping["flags"]))
        parts = energy_breakdown(mapping, tiles, rows_by_flags[flags_key])
        total = sum(parts.values())
        result = {
            "target": TARGET.name, "model": mapping["model"], "cap": mapping["cap"], "node_nm": TARGET.node_nm,
            "tiles_file": path, "latency_ns": mapping["latency_ns"],
            "total_mJ": total / 1e9,
            "parts_mJ": {k: v / 1e9 for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
            "parts_pct": {k: 100 * v / total for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
        }
        out = os.path.join(out_dir, f"cap_{mapping['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        print(f"cap {mapping['cap']}: {total / 1e9:,.3f} mJ per inference -> {out}", flush=True)


if __name__ == "__main__":
    main()
