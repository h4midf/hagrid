"""HyFlexPIM's Table 2 (component area and power of the analog and digital modules),
rebuilt the way the paper built it: the ASADI and ISAAC components it cites, scaled to
HyFlexPIM's configuration and to 65 nm with Stillmaker and Baas. Rows we synthesize
ourselves replace the cited numbers: the ADC (spice/adc/, transistor level), the S&A
(periphery/, the unit Table 2 costs) and the SFU (sfu/).

Reads default/design.json (configuration) and default/components.json (source values
and scaling rules, each with its provenance). Writes default/results/table2.json.
The SFU row is our own synthesis (HyFlexPIM synthesized its SFU itself), read from
sfu/results/sfu.json (sfu/characterize_sfu.py); the digital module sums are given
with and without it.

Usage: python3 targets/hyflexpim/table2.py
"""
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "default"
DESIGN = json.loads((CONFIG / "design.json").read_text())
COMP = json.loads((CONFIG / "components.json").read_text())
sys.path.insert(0, str(HERE / "sfu"))
from characterize_sfu import factors as sfu_factors  # noqa: E402  (45 nm synthesis -> 65 nm)


def area_factor(scaling):
    """Stillmaker and Baas Sec. 4.1: geometric mean of the three area ratios of Table 3."""
    src = scaling["area_geometry"][str(scaling["from"]["node_nm"])]
    dst = scaling["area_geometry"][str(scaling["to"]["node_nm"])]
    feature = (dst[0] / src[0]) ** 2
    pitch = (dst[1] / src[1]) ** 2
    gate = dst[2] / src[2]
    return (feature * pitch * gate) ** (1 / 3)


def energy_factor(scaling):
    """Stillmaker and Baas Eq. (6), EnergyFactor(to) / EnergyFactor(from)."""
    def ef(key, v):
        a2, a1, a0 = scaling["energy_coefficients"][key]
        return a2 * v * v + a1 * v + a0
    src, dst = scaling["from"], scaling["to"]
    return (ef(f"{dst['node_nm']}_{dst['type']}", dst["vdd_V"])
            / ef(f"{src['node_nm']}_HP", src["vdd_V"]))


def s_and_a(units, module):
    """S&A row from our synthesis of the unit Table 2 costs (components.json "s_and_a"),
    scaled 45 nm -> 65 nm with the SFU's Stillmaker factors, at the module's clock."""
    cfg = COMP["s_and_a"]
    path = HERE / cfg["results"]
    if not path.exists():
        raise FileNotFoundError(f"{path}: run periphery/characterize_shift_add.py first")
    unit = next(r for r in json.loads(path.read_text()) if r["unit"] == cfg["circuit"])
    f = sfu_factors()
    clock = cfg["clock_GHz"][module]
    area_um2 = unit["area_um2_45nm"] * f["area"]
    dynamic_mW = unit["dynamic_mW_est"] * clock * f["energy"]
    leakage_mW = unit["leakage_uW"] * 1e-3 * f["power"]
    return {"area_mm2": area_um2 * units * 1e-6, "power_mW": (dynamic_mW + leakage_mW) * units,
            "dynamic_mW": dynamic_mW * units, "leakage_mW": leakage_mW * units, "power_split": "synthesis"}


def adc(units):
    """ADC row: our transistor-level ADC (spice/adc/characterize_adc.py) x units."""
    path = CONFIG / "results" / "adc.json"
    if not path.exists():
        raise FileNotFoundError(f"{path}: run spice/adc/characterize_adc.py first")
    a = json.loads(path.read_text())["adc"]
    # energy per conversion x rate: what the conversions draw, as 1F1C's ADC energy
    return {"area_mm2": a["area_um2"] * units * 1e-6, "power_mW": a["power_mW"] * units,
            "dynamic_mW": a["power_mW"] * units, "power_split": "spice_per_conversion"}


def register(module, reg, fa, fp):
    """IR / OR row: ASADI's register scaled linearly to the HyFlexPIM size and node; the
    OR's power x or_power_extra (components.json rules.ir_or, rules.or_power_extra)."""
    base = COMP["asadi_table1"][module][reg]
    size = DESIGN[module]["registers"][{"ir": "input_register_bytes", "or": "output_register_bytes"}[reg]]
    extra = COMP["rules"]["or_power_extra"] if reg == "or" else 1.0
    return row(base, size / base["size_bytes"], fa, fp, extra_power=extra)


def row(base, ratio, fa, fp, extra_area=1.0, extra_power=1.0):
    """A row scaled from a cited source, whose power is one total (dynamic and leakage
    together): dynamic_mW is that total."""
    power = base["power_mW"] * ratio * fp * extra_power
    return {"area_mm2": base["area_mm2"] * ratio * fa * extra_area,
            "power_mW": power, "dynamic_mW": power, "power_split": "source_total"}


def analog_module(fa, fp):
    cfg = DESIGN["analog_module"]
    base = COMP["asadi_table1"]["analog_module"]
    counts = COMP["hyflexpim_counts"]["analog_module"]
    dac = COMP["isaac_table1"]["dac"]
    arrays = cfg["arrays"]
    cells = arrays * cfg["array"]["rows"] * cfg["array"]["cols"]
    b_arr = base["rram_array"]
    return {
        "rram_array": row(b_arr, cells / (b_arr["arrays"] * b_arr["rows"] * b_arr["cols"]), fa, fp),
        "ir": register("analog_module", "ir", fa, fp),
        "or": register("analog_module", "or", fa, fp),
        "wl_drv": row(dac, arrays * cfg["input"]["wl_drivers_per_array"] / dac["count"], fa, fp),
        "adc": adc(counts["adc"]),
        "s_and_a": s_and_a(counts["s_and_a"], "analog_module"),
        "s_and_h": row(COMP["isaac_table1"]["s_and_h"],
                       counts["s_and_h"] / COMP["isaac_table1"]["s_and_h"]["count"], fa, fp),
    }


def digital_module(fa, fp):
    cfg = DESIGN["digital_module"]
    base = COMP["asadi_table1"]["digital_module"]
    counts = COMP["hyflexpim_counts"]["digital_module"]
    dac = COMP["isaac_table1"]["dac"]
    arrays = cfg["arrays"]
    cells = arrays * cfg["array"]["rows"] * cfg["array"]["cols"]
    b_arr = base["rram_array"]
    # power per cell is ASADI's analog array's (components.json rules.rram_array_digital_power)
    b_pow = COMP["asadi_table1"]["analog_module"]["rram_array"]
    arr = row(b_arr, cells / (b_arr["arrays"] * b_arr["rows"] * b_arr["cols"]), fa, fp)
    arr["power_mW"] = arr["dynamic_mW"] = row(b_pow, cells / (b_pow["arrays"] * b_pow["rows"] * b_pow["cols"]),
                                              fa, fp)["power_mW"]
    return {
        "rram_array": arr,
        "ir": register("digital_module", "ir", fa, fp),
        "or": register("digital_module", "or", fa, fp),
        # one WL driver per row of every array
        "wl_drv": row(dac, arrays * cfg["array"]["rows"] / dac["count"], fa, fp),
        "s_and_a": s_and_a(counts["s_and_a"], "digital_module"),
        "s_and_h": row(COMP["isaac_table1"]["s_and_h"],
                       counts["s_and_h"] / COMP["isaac_table1"]["s_and_h"]["count"], fa, fp),
    }


def sfu_row():
    """The SFU at 65 nm: area from sfu/results/sfu.json (characterize_sfu.py), power the
    average under the paper's workload from sfu/results/sfu_average_power.json
    (power_sfu.py, sfu_average_power.py); None if they have not been run."""
    area_path = HERE / "sfu" / "results" / "sfu.json"
    power_path = HERE / "sfu" / "results" / "sfu_average_power.json"
    if not (area_path.exists() and power_path.exists()):
        return None
    lane = next(r for r in json.loads(area_path.read_text()) if r["unit"] == "sfu_lane")
    s = lane["sfu_65nm"]
    p = json.loads(power_path.read_text())["per_sfu"]
    return {"area_mm2": s["area_mm2"], "power_mW": p["average_mW"],
            "dynamic_mW": p["dynamic_mW"], "leakage_mW": p["leakage_mW"], "power_split": "synthesis",
            "busy_fraction": p["exp_bound_busy_fraction"], "all_busy_mW": s["power_mW"],
            "stage_ps": s["stage_ps"], "meets_clock": s["meets_clock"]}


def summed(components, modules):
    area = math.fsum(c["area_mm2"] for c in components.values() if c is not None)
    power = math.fsum(c["power_mW"] for c in components.values() if c is not None)
    return {"components": components,
            "module_sum": {"area_mm2": area, "power_mW": power},
            "total": {"modules": modules, "area_mm2": area * modules, "power_mW": power * modules}}


def main():
    scaling = COMP["node_scaling"]
    fa, fp = area_factor(scaling), energy_factor(scaling)
    pu = DESIGN["processing_unit"]
    digital = digital_module(fa, fp)
    out = {
        "node_scaling": {"area_factor": fa, "power_factor": fp},
        "analog_module": summed(analog_module(fa, fp), pu["analog_modules"]),
        "digital_module_without_sfu": summed(dict(digital), pu["digital_modules"]),
    }
    sfu = sfu_row()
    if sfu is not None:
        out["digital_module"] = summed({**digital, "sfu": sfu}, pu["digital_modules"])
    path = CONFIG / "results" / "table2.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"area factor {fa:.4f}, power factor {fp:.4f}; wrote {path}")


if __name__ == "__main__":
    main()
