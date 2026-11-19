"""Energy of one inference of a mapped RRAM/SRAM ensemble app, per part.

Runs on a mapping (results/rse_greedy/<app>/cap_<cap>.json) and its placement
(results/rse_tiles/<app>/cap_<cap>.json); it never maps, searches or places. Event
counts follow each op's hardware round in compiler/xdsl_ensemble_timing.py, so energy
and latency count the same events; per-event energies come from TARGET.costs, the
accelerator units from xdsl_accel_concurrency.per_kind_accel_rows of the app's block
(rebuilt with the mapping's own flags), x TARGET.cmos_energy_factor.

Dynamic energy per op (every row / element it processes):
  matvec (every kind)         per row: the input row read from the SRAM (the WMU), written
                              into the PE input register and read out bit-serially
                              (256-bit accesses)
  matvec, stored rhs          per input bit: an RRAM MAC on every macro of a copy's tiles
                              (rram_mac_macro_pJ each, plus rram_mac_column_pJ per used
                              column); the SRAM correction (correction_plan): each of its
                              weights is a PE step, its weight_bits reads (one bit of a
                              read word per PE) and cycles_per_weight logic cycles, shared
                              by the macro's pe_count PEs (its macros are written once,
                              before inference, like the RRAM weights)
  matvec, dynamic rhs         per input bit: every rhs value's slices are PE steps; per
                              copy (pf), the rhs written into its SRAM macros
  lut_search                  per lookup: an RRAM read (one row on) of the table entry's
                              columns, the output written into the SRAM
  accelerator ops             SRAM reads of the dynamic inputs, one circuit evaluation per
                              element / reduction pass / softmax stage step, an SRAM write
                              of the output
  layout ops                  none (reinterpretations, as in the timing model)
An SRAM read or write moves one read / write word per access (TARGET.sram).
Data movement per access, from the placed traffic: router flit-hops and tile bus
transfers (ISAAC's full-activity power / its clock per access, TARGET.organization).
Dynamic energy only: leakage is not counted.

Writes results/rse_energy/<app>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.rse_energy --app resnet50
"""
import argparse
import glob
import json
import math
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "rram-sram-ensemble/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import (
    _TOKEN_ROUND_SUB_KIND, _TOKEN_SCALE_SUB_KIND, _softmax_stage_sub_kind, per_kind_accel_rows,
)
from compiler.xdsl_timing import ABSMAX_TREE_INPUTS
from custom_op_library.common.tools import load_results
from run_scripts.rse_greedy_sweep import RESULTS_DIR
from run_scripts.rse_tiles import TILES_DIR, stored_correction
from targets.target_loader import ensemble_for_app
from targets.target_support import TARGET

ENERGY_DIR = os.path.join(RESULTS_DIR, "rse_energy")

_LAYOUT_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape", "cim.transpose", "cim.unfold"}
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}
# accelerator ops evaluated once per output element
_ELEMENTWISE_OPS = {"cim.vec_add", "cim.vec_el_mul", "cim.vec_select", "cim.gelu", "cim.relu", "cim.silu",
                    "cim.elu", "cim.leaky_relu", "cim.quantize", "cim.dequantize", "cim.rsqrt"}


class _Energy:
    """Energy (pJ) by part of one mapping."""

    def __init__(self, unit_rows: dict):
        self.c = TARGET.costs
        self.s = TARGET.sram
        self.unit_rows = unit_rows
        self.pJ: dict[str, float] = {}

    def add(self, part: str, pj: float):
        self.pJ[part] = self.pJ.get(part, 0.0) + pj

    def sram_read(self, part: str, elements: int, bits: int, times: float = 1):
        self.add(part, times * -(-elements * bits // self.s.read_word_bits) * self.c.sram_read_pJ)

    def sram_write(self, part: str, elements: int, bits: int, times: float = 1):
        self.add(part, times * -(-elements * bits // self.s.write_word_bits) * self.c.sram_write_pJ)

    def pe_steps(self, part: str, steps: float):
        """steps PE weight steps: its reads (one bit of a read word per PE) and logic cycles,
        shared by a macro's PEs."""
        s, c = self.s, self.c
        per_step = (s.weight_bits // s.read_bits_per_pe * c.sram_read_pJ
                    + s.cycles_per_weight * c.sram_logic_cycle_pJ) / s.pe_count
        self.add(part, steps * per_step)

    def evaluations(self, sub_kind: str, count: int):
        if sub_kind not in self.unit_rows:
            raise KeyError(f"no row for accelerator unit {sub_kind!r}")
        self.add(f"accelerator {sub_kind}",
                 count * self.unit_rows[sub_kind]["dynamic_mW_est"] * TARGET.cmos_energy_factor)


def _shape(mapping, ops, value_id) -> list[int]:
    if value_id in ops:
        return ops[value_id]["shape"]
    for a in mapping["args"]:
        if a["id"] == value_id:
            return a["shape"]
    raise KeyError(f"value {value_id} not found")


def _is_weight(ops, value_id) -> bool:
    return value_id in ops and ops[value_id]["op"] in _WEIGHT_OPS


def _matvec(en: _Energy, target, o, mapping, ops):
    lhs, rhs = o["inputs"]
    lhs_shape = _shape(mapping, ops, lhs["from"])
    rhs_shape = _shape(mapping, ops, rhs["from"])
    if len(rhs_shape) == 3:  # grouped: one round per copy covers all blocks
        rows, row_size = math.prod(lhs_shape[:-2]), lhs_shape[-2] * lhs_shape[-1]
    else:
        rows, row_size = math.prod(lhs_shape[:-1]), lhs_shape[-1]
    input_bits = lhs["bits"]
    macs = rows * input_bits
    en.sram_read("SRAM: matvec input reads", row_size, input_bits, rows)
    # each input row is written into its PE's input register and read out bit-serially
    org = TARGET.organization
    ir_bits = org.pe_input_register_access_bits
    ir_accesses = rows * (-(-(row_size * input_bits) // ir_bits) + input_bits * -(-row_size // ir_bits))
    en.add("PE input registers", ir_accesses * org.pe_input_register_access_pJ)
    c, s = en.c, en.s
    if "stationary" in o:
        for st in o["stationary"]:
            columns = st["full_arrays_per_copy"] * TARGET.rram.array.cols + st["partial_tiles_per_copy"] * st["partial_tile_cols"]
            macros = st["full_arrays_per_copy"] + st["partial_tiles_per_copy"]
            en.add("RRAM MAC: macros (WL, decoder, mux select, shift-and-add)", macs * macros * c.rram_mac_macro_pJ)
            en.add("RRAM MAC: columns (ADC, column supply)", macs * columns * c.rram_mac_column_pJ)
        plan = stored_correction(target, o, ops)
        en.pe_steps("SRAM correction: PE steps", macs * plan.weights)
        return
    if _is_weight(ops, rhs["from"]):
        raise RuntimeError(f"op {o['id']}: weight rhs without a stationary placement")
    # dynamic rhs [..., K, N] on the PEs, each value in slices of weight_bits
    slices = -(-rhs["bits"] // s.weight_bits)
    values = math.prod(rhs_shape)
    # per input row, its own block [K, N] of the rhs
    per_row = values // math.prod(rhs_shape[:-2]) if len(rhs_shape) > 2 else values
    en.pe_steps("SRAM dynamic matvec: PE steps", macs * per_row * slices)
    en.sram_write("SRAM dynamic matvec: rhs writes", values * slices, s.weight_bits, o["pf"])


def _lut_search(en: _Energy, o, ops):
    index = [i for i in o["inputs"] if not _is_weight(ops, i["from"])]
    if len(index) != 1:
        raise RuntimeError(f"op {o['id']} (lut_search): expected one dynamic input, got {len(index)}")
    lookups = index[0]["elements"]
    entry = math.prod(o["shape"]) // lookups
    columns = entry * -(-o["bits"] // TARGET.rram.array.bits_per_cell)
    macros = -(-columns // TARGET.rram.array.cols)
    en.add("RRAM read (LUT)", lookups * (macros * en.c.rram_read_macro_pJ + columns * en.c.rram_read_column_pJ))
    en.sram_write("SRAM: op output writes", entry, o["bits"], lookups)


def _accelerator(en: _Energy, o, ops):
    kind = o["op"]
    for inp in o["inputs"]:
        if not _is_weight(ops, inp["from"]):
            en.sram_read("SRAM: op input reads", inp["elements"], inp["bits"])
    numel = math.prod(o["shape"])
    en.sram_write("SRAM: op output writes", numel, o["bits"])
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
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        passes = -(-reduce_size // o["attributes"]["argmax_width"])
        en.evaluations(sub_kind, numel * passes)
    elif kind == "cim.row_absmax":
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        en.evaluations(sub_kind, numel * -(-reduce_size // ABSMAX_TREE_INPUTS))
    elif kind == "cim.quantize_per_token":
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
    target = ensemble_for_app(TARGET, mapping["app"])
    en = _Energy(unit_rows)
    ops = {o["id"]: o for o in mapping["ops"]}
    for o in mapping["ops"]:
        kind = o["op"]
        if kind in _LAYOUT_OPS or kind in _WEIGHT_OPS:
            continue
        if kind == "cim.MatvecOp":
            _matvec(en, target, o, mapping, ops)
        elif kind == "cim.lut_search":
            _lut_search(en, o, ops)
        elif "units" in o:
            _accelerator(en, o, ops)
        else:
            raise NotImplementedError(f"op {o['id']} ({kind}): no energy rule")

    # data movement, per access, from the placed traffic (rse_tiles.py)
    org = TARGET.organization
    traffic = tiles["traffic"]
    en.add("routers", traffic["flit_hops"] * org.router_flit_hop_pJ)
    en.add("tile buses", -(-traffic["bus_bits"] // org.tile_bus_bits) * org.tile_bus_transfer_pJ)
    return en.pJ


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--app", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.app, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no tile results in {os.path.join(TILES_DIR, args.app)}")
    out_dir = os.path.join(ENERGY_DIR, args.app)
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
            "target": TARGET.name, "app": mapping["app"], "model": mapping["model"], "cap": mapping["cap"],
            "node_nm": TARGET.node_nm, "tiles_file": path, "latency_ns": mapping["latency_ns"],
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
