"""Macro efficiency (TOPS/W) from the simulated column and ADC energies.

Accounting (inferred from Laleni's numbers; the paper gives no formula, but this one
reproduces their 111.6 TOPS/W from their own ADC energy and bitline charging):
  per column, per conversion: rows MACs = 2*rows ops
  energy per conversion = ADC energy + column (bitline) charging energy
  TOPS/W = ops / energy
Not included (Laleni does not include them either): wordline drivers, reset device,
digital controller, shift-and-add.

Reads:  results/column.json  (bl_charging, from spice/column/column_sim.py)
        results/adc.json     (enob, from spice/adc/run_enob.py)
Writes: results/efficiency.json

Usage: python3 targets/1fefet1c/compute_efficiency.py
"""
from f1c_io import DESIGN, LALENI, load_result, save_result


def main():
    col = load_result("column")["bl_charging"]
    adc = load_result("adc")["enob"]
    rows = DESIGN["array"]["rows"]
    ops = 2 * rows
    e_col = col["energy_per_column_per_mac_fJ"] * 1e-15
    e_adc = adc["energy_per_conversion_pJ"] * 1e-12
    tops_w = ops / (e_adc + e_col) / 1e12

    # same formula on Laleni's own ADC energy (Walden FoM * 2^ENOB), ADC only
    e_adc_laleni = LALENI["adc_walden_fom_fJ"] * 1e-15 * 2 ** LALENI["adc_enob_bits"]
    data = {
        "ops_per_conversion": ops,
        "adc_energy_pJ": e_adc * 1e12, "column_energy_pJ": e_col * 1e12,
        "column_share_pct": 100 * e_col / (e_adc + e_col),
        "tops_per_w": tops_w,
        "laleni_tops_per_w": LALENI["efficiency_TOPS_per_W"],
        "laleni_adc_energy_from_fom_pJ": e_adc_laleni * 1e12,
        "laleni_implied_column_energy_pJ":
            (ops / (LALENI["efficiency_TOPS_per_W"] * 1e12) - e_adc_laleni) * 1e12,
        "formula": "2*rows / (E_adc + E_column) per column per conversion",
    }
    path = save_result("efficiency", "macro", data)
    print(f"{tops_w:.1f} TOPS/W (Laleni {LALENI['efficiency_TOPS_per_W']}): ADC {e_adc * 1e12:.3f} pJ + "
          f"column {e_col * 1e12:.3f} pJ per conversion -> {path}")


if __name__ == "__main__":
    main()
