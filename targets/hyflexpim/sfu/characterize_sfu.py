"""Synthesizes the HyFlexPIM SFU (rtl/sfu_lane.v and each of its units) with the shared
yosys / NanGate45 flow, with ABC buffering and gate sizing (synth.synth(sized=True)),
and writes results/sfu.json: per circuit, area / critical path / leakage / dynamic
power at 45 nm (custom_op_library/common/tools/report.py), and for the lane the whole
SFU at HyFlexPIM's 65 nm: lanes x area and power, scaled with Stillmaker and Baas
(default/components.json "node_scaling" and "sfu"), and the 1 GHz check.

  area   lane area x lanes x AreaFactor(45 nm HP -> 65 nm bulk)
  power  (dynamic estimate x EnergyFactor + leakage x PowerFactor) x lanes, at 1 GHz;
         the dynamic estimate is report.py's 0.5 alpha C V^2 f (alpha = 0.2, 1.1 V, 1 GHz)
  clock  (critical path + flop overhead) x DelayFactor <= 1 / freq at 65 nm

Usage (from the repo root):
  python3 targets/hyflexpim/sfu/characterize_sfu.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools"))
import report  # noqa: E402
import synth  # noqa: E402

COMP = json.load(open(os.path.join(HERE, "..", "default", "components.json")))
SCALING = COMP["node_scaling"]
SFU = COMP["sfu"]

BASE = ["sfu_pipe", "sfu_prefix_add", "sfu_csa_mul"]
UNITS = {   # circuit -> (rtl modules, instances in a lane, what it computes)
    "sfu_fp16_add": (BASE + ["sfu_fp16_add"], 2, "FP16 add and (second instance) subtract"),
    "sfu_fp16_mul": (BASE + ["sfu_fp16_mul"], 1, "FP16 multiply"),
    "sfu_fp16_div": (BASE + ["sfu_fp16_div"], 1, "FP16 divide, restoring, 1 bit per stage"),
    "sfu_fp16_sqrt": (BASE + ["sfu_fp16_sqrt"], 1, "FP16 square root, restoring, 1 bit per stage"),
    "sfu_fp16_max": (BASE + ["sfu_fp16_max"], 1, "FP16 max (one max-search step)"),
    "sfu_fp16_to_int8": (["sfu_fp16_to_int8"], 1, "FP16 -> int8"),
    "sfu_int8_to_fp16": (["sfu_int8_to_fp16"], 1, "int8 -> FP16 (exp range reduction)"),
    "sfu_fp16_ldexp": (["sfu_fp16_ldexp"], 1, "FP16 x 2^n (exp range reduction)"),
}
LANE = BASE + ["sfu_fp16_add", "sfu_fp16_mul", "sfu_fp16_div", "sfu_fp16_sqrt", "sfu_fp16_max",
               "sfu_fp16_to_int8", "sfu_int8_to_fp16", "sfu_fp16_ldexp", "sfu_lane"]


def poly(coeffs, v):
    out = 0.0
    for c in coeffs:
        out = out * v + c
    return out


def factors():
    """Stillmaker and Baas factors from the synthesis point to HyFlexPIM's 65 nm point."""
    src, dst = SFU["synthesis"], SFU["target"]
    s_key, d_key = f"{src['node_nm']}_{src['type']}", f"{dst['node_nm']}_{dst['type']}"
    g_src = SCALING["area_geometry"][str(src["node_nm"])]
    g_dst = SCALING["area_geometry"][str(dst["node_nm"])]
    area = ((g_dst[0] / g_src[0]) ** 2 * (g_dst[1] / g_src[1]) ** 2 * (g_dst[2] / g_src[2])) ** (1 / 3)

    def ratio(table):
        return (poly(SCALING[table][d_key], dst["vdd_V"]) / poly(SCALING[table][s_key], src["vdd_V"]))
    return {"area": area, "delay": ratio("delay_coefficients"),
            "energy": ratio("energy_coefficients"), "power": ratio("power_coefficients")}


def files(modules):
    return [os.path.join(HERE, "rtl", m + ".v") for m in modules]


def main():
    f = factors()
    budget_ps = 1e6 / SFU["freq_MHz"] / f["delay"] - SFU["flop_overhead_ps_45nm"]
    print(f"factors 45 -> 65 nm: area {f['area']:.4f}, delay {f['delay']:.4f}, energy {f['energy']:.4f}, "
          f"power {f['power']:.4f}; logic per stage at 45 nm <= {budget_ps:.1f} ps", flush=True)
    rows = []
    for unit, (modules, count, what) in UNITS.items():
        res = synth.synth(files(modules), unit, {}, sized=True)
        flops = sum(n for c, n in res["cells"].items() if c.startswith("DFF"))
        row = report.row(unit, {"instances_per_lane": count}, what, res, f"{flops} flops")
        rows.append(row)
        print(f"  {unit}: {res['area_um2']:.0f} um^2, {res['delay_ps']} ps, {flops} flops", flush=True)

    res = synth.synth(files(LANE), "sfu_lane", {}, sized=True)
    flops = sum(n for c, n in res["cells"].items() if c.startswith("DFF"))
    lane = report.row("sfu_lane", {}, "one SFU lane: every unit, outputs registered", res, f"{flops} flops")
    lanes = SFU["lanes"]
    dyn_mW = lane["dynamic_mW_est"] * lanes * f["energy"]
    leak_mW = lane["leakage_uW"] * 1e-3 * lanes * f["power"]
    stage_ps_65 = (res["delay_ps"] + SFU["flop_overhead_ps_45nm"]) * f["delay"]
    lane["sfu_65nm"] = {
        "lanes": lanes,
        "area_mm2": res["area_um2"] * lanes * f["area"] * 1e-6,
        "dynamic_mW": dyn_mW,
        "leakage_mW": leak_mW,
        "power_mW": dyn_mW + leak_mW,
        "stage_ps": stage_ps_65,
        "max_freq_MHz": 1e6 / stage_ps_65,
        "meets_clock": stage_ps_65 <= 1e6 / SFU["freq_MHz"],
        "factors": f,
    }
    rows.append(lane)
    report.save(HERE, "sfu", rows)
    s = lane["sfu_65nm"]
    print(f"  sfu_lane: {res['area_um2']:.0f} um^2, {res['delay_ps']} ps, {flops} flops\n"
          f"  SFU at 65 nm ({lanes} lanes): {s['area_mm2']:.3f} mm^2, {s['power_mW']:.1f} mW "
          f"(dynamic {s['dynamic_mW']:.1f}, leakage {s['leakage_mW']:.1f}), stage {s['stage_ps']:.0f} ps "
          f"-> {s['max_freq_MHz']:.0f} MHz", flush=True)


if __name__ == "__main__":
    main()
