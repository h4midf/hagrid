"""Synthesizes rtl/hfp_shift_add.v (one analog array's S&A) with the shared yosys /
NanGate45 flow, with ABC buffering and gate sizing, and writes results/shift_add.json:
the circuit at 45 nm (custom_op_library/common/tools/report.py) and, scaled with
Stillmaker and Baas to HyFlexPIM's 65 nm (sfu/characterize_sfu.factors), one unit and
the analog module's units, to compare with Table 2's S&A row.

  clock  one code per ADC conversion: design.json adc_sample_rate_GSps (1.28 GHz);
         (critical path + flop overhead) x DelayFactor <= 1 / clock at 65 nm
  area   unit area x AreaFactor
  power  (dynamic estimate x clock / 1 GHz x EnergyFactor + leakage x PowerFactor);
         report.py's dynamic estimate is 0.5 alpha C V^2 at 1 GHz (alpha = 0.2, 1.1 V)

Also synthesizes rtl/isaac_shift_add.v (a bare 16-bit shift-and-add, the unit ISAAC's
Table I sizes) and scales it to ISAAC's 32 nm, 1.2 GHz, to check the flow against the
source of Table 2's S&A row (ISAAC S+A: 0.00006 mm^2, 0.05 mW).

Usage (from the repo root):
  python3 targets/hyflexpim/periphery/characterize_shift_add.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
sys.path.insert(0, os.path.join(HERE, "..", "sfu"))
import report  # noqa: E402
import synth  # noqa: E402
from characterize_sfu import SCALING, SFU, factors, poly  # noqa: E402

DESIGN = json.load(open(os.path.join(HERE, "..", "default", "design.json")))
MODULE = "hfp_shift_add"
PREFIX_ADD = os.path.join(HERE, "..", "sfu", "rtl", "sfu_prefix_add.v")
FILES = [os.path.join(HERE, "rtl", MODULE + ".v"), PREFIX_ADD]

# ISAAC Table I, IMA: S+A 'number 4, 0.2 mW, 0.00024 mm^2' at 32 nm, 1.2 GHz
ISAAC_SA = {"area_um2": 60.0, "power_mW": 0.05, "node_nm": 32, "clock_GHz": 1.2, "vdd_V": 1.0}


def isaac_check():
    """Our bare 16-bit shift-and-add at ISAAC's 32 nm (HP, 1.0 V as in components.json's
    32 nm point) and 1.2 GHz, against ISAAC's S+A row."""
    src = SFU["synthesis"]
    g45 = SCALING["area_geometry"][str(src["node_nm"])]
    g32 = SCALING["area_geometry"]["32"]
    area_f = ((g32[0] / g45[0]) ** 2 * (g32[1] / g45[1]) ** 2 * (g32[2] / g45[2])) ** (1 / 3)
    energy_f = (poly(SCALING["energy_coefficients"]["32_HP"], ISAAC_SA["vdd_V"])
                / poly(SCALING["energy_coefficients"][f"{src['node_nm']}_{src['type']}"], src["vdd_V"]))
    res = synth.synth([os.path.join(HERE, "rtl", "isaac_shift_add.v"), PREFIX_ADD], "isaac_shift_add", {},
                      sized=True)
    row = report.row("isaac_shift_add", {"W": 16}, "bare 16-bit shift-and-add (ISAAC's S+A)", res,
                     "combinational, no registers")
    area_32 = res["area_um2"] * area_f
    power_32 = row["dynamic_mW_est"] * ISAAC_SA["clock_GHz"] * energy_f
    row["isaac_32nm"] = {"area_um2": area_32, "power_mW": power_32,
                         "isaac_area_um2": ISAAC_SA["area_um2"], "isaac_power_mW": ISAAC_SA["power_mW"]}
    print(f"  isaac_shift_add: {res['area_um2']:.0f} um^2 (45 nm) -> {area_32:.1f} um^2, {power_32:.4f} mW "
          f"at 32 nm, 1.2 GHz (ISAAC: {ISAAC_SA['area_um2']:.0f} um^2, {ISAAC_SA['power_mW']} mW)", flush=True)
    return row


def main():
    f = factors()
    clock_GHz = DESIGN["analog_module"]["readout"]["adc_sample_rate_GSps"]
    period_ps = 1e3 / clock_GHz
    budget_ps = period_ps / f["delay"] - SFU["flop_overhead_ps_45nm"]
    units = DESIGN["analog_module"]["arrays"]          # one S&A per array (Table 2: 512)

    res = synth.synth(FILES, MODULE, {}, sized=True)
    flops = sum(n for c, n in res["cells"].items() if c.startswith("DFF"))
    row = report.row(MODULE, {"PART": 14, "ACC": 22}, "S&A of one analog array: carry-save over a "
                     "weight's columns, then partial add and accumulator update (OR outside)", res,
                     f"{flops} flops")
    dyn_mW = row["dynamic_mW_est"] * clock_GHz * f["energy"]
    leak_mW = row["leakage_uW"] * 1e-3 * f["power"]
    stage_ps = (res["delay_ps"] + SFU["flop_overhead_ps_45nm"]) * f["delay"]
    unit = {"area_um2": res["area_um2"] * f["area"], "dynamic_mW": dyn_mW, "leakage_mW": leak_mW,
            "power_mW": dyn_mW + leak_mW, "stage_ps": stage_ps, "clock_ps": period_ps,
            "meets_clock": stage_ps <= period_ps}
    row["hyflexpim_65nm"] = {
        "clock_GHz": clock_GHz,
        "unit": unit,
        "analog_module": {"units": units, "area_mm2": unit["area_um2"] * units * 1e-6,
                          "power_mW": unit["power_mW"] * units},
        "factors": f,
    }
    report.save(HERE, "shift_add", [row, isaac_check()])
    m = row["hyflexpim_65nm"]["analog_module"]
    print(f"logic per stage at 45 nm <= {budget_ps:.1f} ps (1 / {clock_GHz} GHz at 65 nm)\n"
          f"  {MODULE}: {res['area_um2']:.0f} um^2 (45 nm), {res['delay_ps']} ps, {flops} flops\n"
          f"  one unit at 65 nm: {unit['area_um2']:.0f} um^2, {unit['power_mW']:.3f} mW "
          f"(dynamic {dyn_mW:.3f}, leakage {leak_mW:.3f}), stage {stage_ps:.0f} ps of {period_ps:.0f}\n"
          f"  analog module ({units} units): {m['area_mm2']:.4f} mm^2, {m['power_mW']:.2f} mW", flush=True)


if __name__ == "__main__":
    main()
