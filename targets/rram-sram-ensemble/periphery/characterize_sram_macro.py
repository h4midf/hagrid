"""The SRAM macro: its logic (scaling.sram_logic_rtl(), shared yosys / NanGate45 flow, scaled to
65 nm, 100 MHz) and its 32 x 64 array (CACTI 6.5, targets/hyflexpim/CACTI). Writes
results/sram_macro.json.

  array power = leakage + read energy x read rate; a weight takes 3 read cycles of every
  4 (3 loads, 1 compute).

Usage (from the repo root, CACTI built; see targets/hyflexpim/periphery/cacti_run.py):
  python3 targets/rram-sram-ensemble/periphery/characterize_sram_macro.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "..", "hyflexpim", "periphery"))
sys.path.insert(0, HERE)
import report  # noqa: E402
import synth  # noqa: E402
from cacti_run import run as cacti  # noqa: E402
from scaling import DESIGN, at_chip, sram_logic_rtl  # noqa: E402

CELLS = ("itrs-hp", "itrs-lop", "itrs-lstp")


def array_cost(cell: str, size_bytes: int, access_bytes: int, node_nm: int, read_rate_MHz: float) -> dict:
    r = cacti(size_bytes, access_bytes, node_nm, cell=cell, peripheral=cell)
    if r is None:
        raise RuntimeError(f"CACTI found no organization for {size_bytes} B, {access_bytes} B access, {cell}")
    r["cell"] = cell
    r["read_rate_MHz"] = read_rate_MHz
    r["power_mW"] = r["leakage_mW"] + r["read_energy_pJ"] * read_rate_MHz * 1e-3
    return r


def main() -> None:
    assumed = DESIGN["model_assumptions"]["sram_logic"]
    w = assumed["weight_bits"]
    res = synth.synth([sram_logic_rtl()], "rse_sram_logic", {})
    row = report.row("rse_sram_logic", {"W": w, "ACC": assumed["accumulator_bits"]},
                     "SRAM macro logic: 16 PEs, FSM, input and output buffers", res)
    row["chip_65nm"] = at_chip(row)
    report.save(HERE, "sram_logic", [row])

    sm = DESIGN["sram_macro"]
    size_bytes = sm["array"]["rows"] * sm["array"]["cols"] // 8
    rate = DESIGN["clock"]["freq_MHz"] * w / (w + 1)
    node = DESIGN["process"]["node_nm"]
    arrays = {c: array_cost(c, size_bytes, assumed["access_bytes"], node, rate) for c in CELLS}
    report_rows = [{"unit": "sram_array", "cell": c, **r} for c, r in arrays.items()]
    report.save(HERE, "sram_array", report_rows)

    c = row["chip_65nm"]
    print(f"logic: {res['area_um2']:.0f} um^2 (45 nm) -> {c['area_um2']:.0f} um^2, {c['power_mW']:.4f} mW "
          f"({c['dynamic_mW']:.4f} dynamic + {c['leakage_mW']:.4f} leakage)")
    for name, r in arrays.items():
        print(f"array {name}: {r['area_mm2'] * 1e6:.0f} um^2, read {r['read_energy_pJ']:.3f} pJ, leakage "
              f"{r['leakage_mW']:.4f} mW, {r['power_mW']:.4f} mW at {r['read_rate_MHz']:.0f} MHz reads")


if __name__ == "__main__":
    main()
