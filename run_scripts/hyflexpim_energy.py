"""Dynamic energy of one inference of a mapped HyFlexPIM model, per part.

Runs on a mapping (results/hyflexpim_greedy/<model>/cap_<cap>.json) and its placement
(results/hyflexpim_tiles/<model>/cap_<cap>.json); it never maps, searches or places.
The same sources as hyflexpim_area.py: our Table 2 reproduction for the modules
(TARGET.analog / TARGET.digital components, at full activity), the bandwidth-scaled ISAAC
interconnect, our CMOS units. Event counts follow each op's round in the timing model
(compiler/xdsl_dual_module_timing.py).

Per-event energies:
  analog array cycle  one array, one 100 ns cycle (an input bit, or a LUT row read): the
                      analog module's components' dynamic power but the ADC's / its arrays
                      x the cycle
  ADC conversion      per column converted, only the columns a copy uses (HyFlexPIM Sec. 3.2
                      counts conversions): an SLC column in the ADC's 6-bit mode, an MLC
                      column in its 7-bit mode (TARGET.analog.adc_conversion_pJ_*, our SPICE
                      ADC)
  digital row         one digital array row processed (its write and read cycles): the
                      digital module's components' dynamic power without the SFU / its
                      arrays x the row time; a row read or write alone takes its cycles'
                      share
  A component's dynamic power (ComponentCost.dynamic_mW) is our own split where we model
  it (the S&A from synthesis, the ADC's energy per conversion from SPICE); a row scaled
  from a cited source (RRAM arrays, WL drivers, S&H, registers) and the ISAAC
  interconnect give one total, which is taken whole.
  bus bit             TARGET.inner_bus / global_router power / its bandwidth
  CMOS unit           a circuit evaluation: its dynamic energy (custom_op_library, 1 GHz)
                      x TARGET.cmos_energy_factor

Events per op:
  analog matvec     input rows x input bits x the arrays of a weight copy; the input
                    read from and the output written to the digital module (where the
                    activations and partial sums are kept, HyFlexPIM Sec. 3.3)
  lut_search        lookups x the arrays of the table copy (one cycle each); index read,
                    row written
  digital matvec    outputs x the digital rows of one output on one array (its K
                    multiplies and K - 1 additions, as xdsl_dual_module_timing; the work
                    is the same whatever arrays it spreads over)
  accelerator ops   digital row reads of the dynamic inputs, evaluations as in
                    fefet1c_energy.py, a digital row write of the output
  traffic           the placement's inner-bus and global-bus bits
Dynamic energy only, as fefet1c_energy.py. Layout ops cost nothing.

Writes results/hyflexpim_energy/<model>/cap_<cap>.json.

Usage (from the repo root):
  python3 -m run_scripts.hyflexpim_energy --model bert-base
"""
import argparse
import glob
import json
import math
import os

# Set before the compiler imports: targets.target_support reads it at import time.
os.environ["TARGET_NAME"] = "hyflexpim/default"

from compiler.lowering import build_tiled_cim_block
from compiler.xdsl_accel_concurrency import (
    _TOKEN_ROUND_SUB_KIND, _TOKEN_SCALE_SUB_KIND, _softmax_stage_sub_kind, per_kind_accel_rows,
)
from compiler.xdsl_timing import ABSMAX_TREE_INPUTS
from custom_op_library.common.tools import load_results
from run_scripts.hyflexpim_greedy_sweep import RESULTS_DIR
from run_scripts.hyflexpim_tiles import TILES_DIR
from targets.target_support import TARGET

ENERGY_DIR = os.path.join(RESULTS_DIR, "hyflexpim_energy")
EXCLUDED_DIGITAL = ("sfu",)   # replaced by our CMOS units

_LAYOUT_OPS = {"cim.broadcast", "cim.permute", "cim.view", "cim.reshape", "cim.transpose", "cim.unfold"}
_WEIGHT_OPS = {"const.ref", "cim.save_stationary"}
_ELEMENTWISE_OPS = {"cim.vec_add", "cim.vec_el_mul", "cim.vec_select", "cim.gelu", "cim.relu", "cim.silu",
                    "cim.elu", "cim.leaky_relu", "cim.quantize", "cim.dequantize", "cim.rsqrt"}


class _Energy:
    """Energy (pJ) by part of one mapping."""

    def __init__(self, unit_rows: dict):
        an, dg = TARGET.analog, TARGET.digital
        if (an.adc_bits_slc, an.adc_bits_mlc) != (6, 7):
            raise ValueError(f"the ADC energies are for 6- and 7-bit conversions, not {an.adc_bits_slc} and "
                             f"{an.adc_bits_mlc}")
        self.unit_rows = unit_rows
        # every analog component but the ADC, per array cycle; the ADC per conversion
        self.analog_cycle_pJ = sum(c.dynamic_mW for k, c in an.components.items() if k != "adc") \
            / an.arrays * an.cycle_ns
        self.conversion_pJ = {"slc": an.adc_conversion_pJ_slc, "mlc": an.adc_conversion_pJ_mlc}
        row_ns = dg.row_read_ns + dg.row_write_ns
        digital_mW = sum(c.dynamic_mW for k, c in dg.components.items() if k not in EXCLUDED_DIGITAL)
        self.digital_row_pJ = digital_mW / dg.arrays * row_ns
        self.digital_read_pJ = self.digital_row_pJ * dg.row_read_ns / row_ns
        self.digital_write_pJ = self.digital_row_pJ * dg.row_write_ns / row_ns
        self.inner_bus_bit_pJ = TARGET.inner_bus.power_mW / (TARGET.inner_bus_GB_per_s * 8)
        self.global_bus_bit_pJ = TARGET.global_router.power_mW / (TARGET.global_bus_GB_per_s * 8)
        self.pJ: dict[str, float] = {}

    def add(self, part: str, pj: float):
        self.pJ[part] = self.pJ.get(part, 0.0) + pj

    def digital_rows(self, elements: int, bits: int) -> int:
        """Digital-array rows holding elements x bits (one bit per cell)."""
        return -(-elements * bits // TARGET.digital.array.cols)

    def read(self, elements: int, bits: int, times: float = 1):
        self.add("digital row reads", times * self.digital_rows(elements, bits) * self.digital_read_pJ)

    def write(self, elements: int, bits: int, times: float = 1):
        self.add("digital row writes", times * self.digital_rows(elements, bits) * self.digital_write_pJ)

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


def _copy_arrays(o) -> int:
    """Arrays of one copy of the op's stored operand."""
    (s,) = o["stationary"]
    return s["full_arrays_per_copy"] + s["partial_tiles_per_copy"]


def _copy_columns(o, ops, split: bool) -> dict:
    """{"slc": columns, "mlc": columns} of one copy of the op's stored operand, as
    xdsl_dual_module_hardware._stationary_layout lays it out: per row tile, slc_rate_pct
    of the output columns in SLC and the rest in MLC (split), or all in SLC (a LUT
    table). Checked against the copy's placed columns."""
    (s,) = o["stationary"]
    stored = ops[s["value"]]
    shape, bits, an = stored["shape"], stored["bits"], TARGET.analog
    blocks, in_dim, out_dim = (1, shape[0], 1) if len(shape) == 1 else (math.prod(shape[:-2]), shape[-2], shape[-1])
    strips = blocks * -(-in_dim // an.array.rows)
    slc_out = -(-out_dim * an.slc_rate_pct // 100) if split else out_dim
    cols = {"slc": strips * slc_out * -(-bits // an.slc_bits),
            "mlc": strips * (out_dim - slc_out) * -(-bits // an.mlc_bits)}
    placed = s["full_arrays_per_copy"] * an.array.cols + s["partial_tiles_per_copy"] * s["partial_tile_cols"]
    if cols["slc"] + cols["mlc"] != placed:
        raise RuntimeError(f"op {o['id']}: {cols} columns, but the copy is placed on {placed}")
    return cols


def _conversions_pJ(en, cols: dict) -> float:
    """ADC energy of converting every column once."""
    return sum(n * en.conversion_pJ[mode] for mode, n in cols.items())


def _digital_rows_per_output(k: int, lhs_bits: int, rhs_bits: int) -> int:
    """Digital rows of one output on one array: k int8 x int8 multiplies, k - 1 additions
    of 16 + ceil(log2 k) bits, each NOR taking logic_op_columns of a row
    (xdsl_dual_module_timing._digital_rows_per_output with one array)."""
    d = TARGET.digital
    if (lhs_bits, rhs_bits) != (8, 8):
        raise NotImplementedError(f"digital matvec of {lhs_bits} x {rhs_bits} bits: only int8 x int8 is costed")
    acc_bits = lhs_bits + rhs_bits + math.ceil(math.log2(k))
    mul_per_row = d.array.cols // (d.int8_mul_logic_ops * d.logic_op_columns)
    add_per_row = d.array.cols // (d.add_logic_ops_per_bit * acc_bits * d.logic_op_columns)
    return -(-k // mul_per_row) + -(-(k - 1) // add_per_row)


def _matvec(en: _Energy, o, mapping, ops):
    lhs, rhs = o["inputs"]
    lhs_shape = _shape(mapping, ops, lhs["from"])
    rhs_shape = _shape(mapping, ops, rhs["from"])
    numel = math.prod(o["shape"])
    if "stationary" in o:
        rows = math.prod(lhs_shape[:-2]) if len(rhs_shape) == 3 else math.prod(lhs_shape[:-1])
        rounds = rows * lhs["bits"]
        en.add("analog array cycles", rounds * _copy_arrays(o) * en.analog_cycle_pJ)
        en.add("analog ADC conversions", rounds * _conversions_pJ(en, _copy_columns(o, ops, split=True)))
        en.read(lhs["elements"], lhs["bits"])
        en.write(numel, o["bits"])
        return
    if _is_weight(ops, rhs["from"]):
        raise RuntimeError(f"op {o['id']}: weight rhs without a stationary placement")
    k = lhs_shape[-1]
    en.add("digital matvec rows", numel * _digital_rows_per_output(k, lhs["bits"], rhs["bits"]) * en.digital_row_pJ)


def _lut_search(en: _Energy, o, ops):
    index = [i for i in o["inputs"] if not _is_weight(ops, i["from"])]
    if len(index) != 1:
        raise RuntimeError(f"op {o['id']} (lut_search): expected one dynamic input, got {len(index)}")
    lookups = index[0]["elements"]
    en.add("analog array cycles (LUT)", lookups * _copy_arrays(o) * en.analog_cycle_pJ)
    en.add("analog ADC conversions (LUT)", lookups * _conversions_pJ(en, _copy_columns(o, ops, split=False)))
    en.read(lookups, index[0]["bits"])
    en.write(math.prod(o["shape"]), o["bits"])


def _accelerator(en: _Energy, o, ops):
    kind = o["op"]
    for inp in o["inputs"]:
        if not _is_weight(ops, inp["from"]):
            en.read(inp["elements"], inp["bits"])
    numel = math.prod(o["shape"])
    en.write(numel, o["bits"])
    if kind in _ELEMENTWISE_OPS:
        (sub_kind,) = o["units"]
        en.evaluations(sub_kind, numel)
    elif kind == "cim.reduce_vec":
        if o["attributes"]["reduce_impl"] != "tree":
            raise NotImplementedError(f"op {o['id']}: reduce_impl {o['attributes']['reduce_impl']!r}")
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        en.evaluations(sub_kind, numel * -(-reduce_size // o["attributes"]["reduce_width"]))
    elif kind == "cim.max":
        (sub_kind,) = o["units"]
        reduce_size = o["inputs"][0]["elements"] // numel
        en.evaluations(sub_kind, numel * -(-reduce_size // o["attributes"]["argmax_width"]))
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
            en.evaluations(_softmax_stage_sub_kind(module, row, o["inputs"][0]["bits"]),
                           rows * row_width if per_element else rows * passes)
    elif kind == "cim.layer_norm":
        # the streamed circuit: every cycle of a row (dynamic_mW_est is pJ per cycle)
        (sub_kind,) = o["units"]
        rows = numel // o["shape"][-1]
        en.evaluations(sub_kind, rows * en.unit_rows[sub_kind]["cycles_per_row"])
    else:
        raise NotImplementedError(f"op {o['id']} ({kind}): no energy rule")


def energy_breakdown(mapping: dict, tiles: dict, unit_rows: dict) -> tuple:
    """({part: pJ} of one inference, the per-event energies)."""
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
    traffic = tiles["traffic"]
    en.add("inner buses", traffic["inner_bus_bits"] * en.inner_bus_bit_pJ)
    en.add("routers", traffic["global_bus_bits"] * en.global_bus_bit_pJ)
    events = {"analog_array_cycle_pJ": en.analog_cycle_pJ, "adc_conversion_pJ": en.conversion_pJ,
              "digital_row_pJ": en.digital_row_pJ,
              "inner_bus_bit_pJ": en.inner_bus_bit_pJ, "global_bus_bit_pJ": en.global_bus_bit_pJ}
    return en.pJ, events


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True)
    args = parser.parse_args()

    paths = sorted(glob.glob(os.path.join(TILES_DIR, args.model, "cap_*.json")))
    if not paths:
        raise FileNotFoundError(f"no placements in {os.path.join(TILES_DIR, args.model)}")
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
        parts, events = energy_breakdown(mapping, tiles, rows_by_flags[flags_key])
        total = sum(parts.values())
        result = {
            "target": TARGET.name, "model": mapping["model"], "cap": mapping["cap"], "node_nm": TARGET.node_nm,
            "tiles_file": path, "latency_ns": mapping["latency_ns"], "event_energies": events,
            "total_mJ": total / 1e9,
            "parts_mJ": {k: v / 1e9 for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
            "parts_pct": {k: 100 * v / total for k, v in sorted(parts.items(), key=lambda kv: -kv[1])},
        }
        out = os.path.join(out_dir, f"cap_{mapping['cap']}.json")
        with open(out, "w") as f:
            json.dump(result, f, indent=2)
        top = ", ".join(f"{k} {v:.1f}%" for k, v in list(result["parts_pct"].items())[:3])
        print(f"cap {mapping['cap']}: {total / 1e9:,.3f} mJ per inference ({top}) -> {out}", flush=True)


if __name__ == "__main__":
    main()
