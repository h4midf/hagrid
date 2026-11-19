"""SRAM-over-RRAM area and power per network, to compare with Sec. V-F, Fig. 12.

The macros are fixed, so the ratio differs per network only through how many of each it
needs: macros = memory (Fig. 11b) / one macro's capacity. An RRAM macro holds
rows x cols x the network's RRAM bits per cell (Table III); an SRAM macro rows x cols bits.

  ratio = (SRAM macros x one SRAM macro) / (RRAM macros x one RRAM macro)

Also given: the one-macro ratio each published number implies. RRAM power depends on the
rows on, which the paper does not give, so power is given per rows on.
Writes default/results/sram_over_rram.json.

Usage (from the repo root): python3 targets/rram-sram-ensemble/sram_over_rram.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "default" / "results"
DESIGN = json.loads((HERE / "default" / "design.json").read_text())
PAPER = DESIGN["reported_results"]
ROWS_SHOWN = ("8", "16", "32")


def main() -> None:
    logic = json.loads((HERE / "periphery" / "results" / "sram_logic.json").read_text())[0]["chip_65nm"]
    arrays = {r["cell"]: r for r in json.loads((HERE / "periphery" / "results" / "sram_array.json").read_text())}
    array = arrays[DESIGN["model_assumptions"]["sram_logic"]["cacti_cell"]]
    rram_area = json.loads((RESULTS / "rram_macro_area.json").read_text())["total_um2"]
    rram_power = json.loads((RESULTS / "rram_macro_power.json").read_text())["by_rows_on"]

    sram_area = logic["area_um2"] + array["area_mm2"] * 1e6
    logic_power = json.loads((HERE / "periphery" / "results" / "sram_logic_power_real.json").read_text())["power_mW"]
    array_power = json.loads((RESULTS / "sram_array_spice.json").read_text())["power_mW"]
    sram_power = logic_power + array_power
    rm, sm = DESIGN["rram_macro"]["array"], DESIGN["sram_macro"]["array"]
    sram_bits = sm["rows"] * sm["cols"]

    fig6 = PAPER["fig6_layout_px"]
    layout_ratio = ((fig6["mac_engine"] + fig6["sram_memory_array"])
                    / (fig6["rram_array_and_decoder"] + fig6["read_out_circuit"]))

    networks = {}
    for name, area_pct in PAPER["sram_over_rram_area_pct"].items():
        mem = PAPER["memory_MB"][name]
        rram_bits = rm["rows"] * rm["cols"] * PAPER["accuracy_table3"][name]["rram_bits"]
        count_ratio = (mem["sram"] / sram_bits) / (mem["rram"] / rram_bits)
        power_pct = PAPER["sram_over_rram_power_pct"][name]
        networks[name] = {
            "macro_count_ratio": count_ratio,
            "area_pct": {"paper": area_pct, "ours": 100 * count_ratio * sram_area / rram_area,
                         "from_fig6_layout": 100 * count_ratio * layout_ratio},
            "power_pct": {"paper": power_pct,
                          "ours_by_rows_on": {n: 100 * count_ratio * sram_power / rram_power[n]["total_mW"]
                                              for n in ROWS_SHOWN}},
            "implied_one_macro_area_ratio": area_pct / 100 / count_ratio,
            "implied_one_macro_power_ratio": power_pct / 100 / count_ratio,
        }

    out = {"sram_macro": {"area_um2": sram_area, "logic_area_um2": logic["area_um2"],
                          "array_area_um2": array["area_mm2"] * 1e6, "power_mW": sram_power,
                          "logic_power_mW": logic_power, "array_power_mW": array_power},
           "rram_macro": {"area_um2": rram_area, "power_mW_by_rows_on": {n: rram_power[n]["total_mW"] for n in ROWS_SHOWN}},
           "one_macro": {"area_ratio": sram_area / rram_area, "area_ratio_from_fig6_layout": layout_ratio,
                         "power_ratio_by_rows_on": {n: sram_power / rram_power[n]["total_mW"] for n in ROWS_SHOWN}},
           "networks": networks}
    (RESULTS / "sram_over_rram.json").write_text(json.dumps(out, indent=2) + "\n")

    print(f"SRAM macro: {sram_area:.0f} um^2 (logic {logic['area_um2']:.0f}, array {array['area_mm2'] * 1e6:.0f}), "
          f"{sram_power:.4f} mW (logic {logic_power:.4f}, array {array_power:.4f})")
    print(f"Fig. 6 layout: one-macro area ratio {layout_ratio:.3f}")
    print(f"RRAM macro: {rram_area:.0f} um^2; one-macro area ratio {sram_area / rram_area:.3f}; power ratio "
          + ", ".join(f"{sram_power / rram_power[n]['total_mW']:.4f} ({n} rows)" for n in ROWS_SHOWN))
    for name, v in networks.items():
        p = v["power_pct"]["ours_by_rows_on"]
        print(f"{name:<24} macros x{v['macro_count_ratio']:.3f} | area {v['area_pct']['ours']:5.1f}%, Fig. 6 "
              f"{v['area_pct']['from_fig6_layout']:5.1f}% (paper {v['area_pct']['paper']}%, implies "
              f"{v['implied_one_macro_area_ratio']:.3f}) | power "
              + "/".join(f"{p[n]:.2f}" for n in ROWS_SHOWN)
              + f"% (paper {v['power_pct']['paper']}%, implies {v['implied_one_macro_power_ratio']:.4f})")


if __name__ == "__main__":
    main()
