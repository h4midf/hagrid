"""Area of the RRAM macro at the blocks Sec. V-F lists (crossbar, WL driver and shifter,
decoder, flash ADCs, pMOS headers, BL / SL / column muxes). Transistors use NeuroSim's
single-transistor layout (default/components.json "layout"), capacitors the MOM density,
resistors a poly strip. Writes default/results/rram_macro_area.json.

Usage (from the repo root): python3 targets/rram-sram-ensemble/rram_macro_area.py
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "default" / "results"
DESIGN = json.loads((HERE / "default" / "design.json").read_text())
LAYOUT = json.loads((HERE / "default" / "components.json").read_text())["layout"]
ASSUMED = DESIGN["model_assumptions"]
F_M = DESIGN["process"]["node_nm"] * 1e-9
READOUT_DIR = HERE / "spice" / "readout"


def transistor_um2(w_m: float, l_m: float) -> float:
    lay = LAYOUT["transistor_layout_F"]
    return (2 * (l_m + lay["gate_poly_gap"] * F_M)
            * (w_m + lay["p_n_diffusion_gap"] * F_M + 2 * (lay["poly_extension"] + lay["field_poly_gap"] / 2) * F_M)) * 1e12


def _value(expr: str, env: dict) -> float:
    e = re.sub(r"(\d+(?:\.\d+)?)n\b", r"(\1e-9)", expr.strip("{}"))
    return float(eval(e, {}, env))


def _subckts() -> dict:
    subs = {}
    text = (READOUT_DIR / "flash_cells.lib").read_text()
    for m in re.finditer(r"^\.subckt\s+(\w+)\s+([^\n]*)\n(.*?)^\.ends", text, re.S | re.M):
        params = {t.split("=")[0]: float(t.split("=")[1]) for t in m.group(2).split() if "=" in t}
        subs[m.group(1)] = (params, [x for x in m.group(3).splitlines() if x.strip() and not x.startswith("*")])
    return subs


def _flat_um2(cell: str, env: dict, subs: dict) -> tuple:
    params, body = subs[cell]
    env = {**params, **env}
    area = count = 0
    for line in body:
        t = line.split()
        if t[0][0] in "Mm":
            kv = dict(x.split("=") for x in t[6:] if "=" in x)
            area += transistor_um2(_value(kv["w"], env), _value(kv["l"], env))
            count += 1
        elif t[0][0] in "Xx":
            name = next(x for x in t[1:] if x in subs)
            sub_env = {x.split("=")[0]: float(x.split("=")[1]) for x in t[t.index(name) + 1:] if "=" in x}
            a, n = _flat_um2(name, sub_env, subs)
            area, count = area + a, count + n
    return area, count


def flash_adc_um2() -> dict:
    subs = _subckts()
    transistors = count = 0
    for line in (READOUT_DIR / "flash_tb.sp").read_text().splitlines():
        t = line.split()
        if t and t[0][0] == "X":
            name = next(x for x in t[1:] if x in subs)
            env = {x.split("=")[0]: float(x.split("=")[1]) for x in t[t.index(name) + 1:] if "=" in x}
            a, n = _flat_um2(name, env, subs)
            transistors, count = transistors + a, count + n
    r = ASSUMED["readout"]
    squares = r["ladder_segment_kohm"] * 1e3 / LAYOUT["poly_ohm_per_square"]
    resistors = 8 * squares * LAYOUT["poly_width_um"] ** 2
    caps = 7 * r["ladder_tap_decoupling_fF"] / LAYOUT["mom_density_fF_per_um2"]
    return {"transistors_um2": transistors, "transistor_count": count, "resistors_um2": resistors,
            "capacitors_um2": caps, "total_um2": transistors + resistors + caps}


def main() -> None:
    rm = DESIGN["rram_macro"]
    rows, cols = rm["array"]["rows"], rm["array"]["cols"]
    readouts = rm["readout"]["readout_circuits"]
    wl, mux, thick = ASSUMED["wl"], ASSUMED["column_mux"], ASSUMED["thick_oxide"]
    readout = json.loads((RESULTS / "readout.json").read_text())
    decoder = json.loads((HERE / "periphery" / "results" / "rram_decoder.json").read_text())[0]["chip_65nm"]
    n, l_hv = 1e-9, thick["l_nm"] * 1e-9
    l_core = 65e-9

    gate_um2 = transistor_um2(mux["nmos_w_nm"] * n, l_core) + transistor_um2(mux["pmos_w_nm"] * n, l_core)
    select_driver_um2 = 2 * (transistor_um2(2080 * n, l_core) + transistor_um2(1040 * n, l_core))
    wl_unit = (2 * transistor_um2(520 * n, l_core) + 2 * transistor_um2(260 * n, l_core)
               + 2 * transistor_um2(wl["level_shifter_nmos_w_nm"] * n, l_hv)
               + 2 * transistor_um2(wl["level_shifter_pmos_w_nm"] * n, wl["level_shifter_pmos_l_nm"] * n)
               + transistor_um2(1000 * n, l_hv) + transistor_um2(500 * n, l_hv)
               + transistor_um2(wl["driver_pmos_w_nm"] * n, l_hv) + transistor_um2(wl["driver_nmos_w_nm"] * n, l_hv))
    adc = flash_adc_um2()
    parts = {
        "crossbar": rows * cols * ASSUMED["array_layout"]["cell_pitch_um"] ** 2,
        "flash_adcs": readouts * adc["total_um2"],
        "headers": readouts * transistor_um2(readout["header"]["width_nm"] * n, l_core),
        "column_mux": cols * gate_um2 + rm["readout"]["column_mux"] * select_driver_um2,
        "programming_muxes": (rows + cols) * gate_um2,
        "wl_drivers_and_shifters": rows * wl_unit,
        "decoder": decoder["area_um2"],
    }
    total = sum(parts.values())
    out = {"flash_adc_each": adc, "parts_um2": parts, "total_um2": total}
    (RESULTS / "rram_macro_area.json").write_text(json.dumps(out, indent=2) + "\n")
    for k, v in sorted(parts.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<26} {v:10.0f} um^2  {100 * v / total:5.1f}%")
    print(f"  {'total':<26} {total:10.0f} um^2 (one ADC: {adc['transistor_count']} transistors, {adc['total_um2']:.0f} um^2)")


if __name__ == "__main__":
    main()
