"""The RRAM macro's read-out over a sweep of the read supply (above the pMOS header), with the
ADC and logic staying at 1.2 V. Writes default/results/read_voltage_sweep.json.

Per read voltage:
  header    the pMOS width reaching the target resistance, or none if the header cannot
            turn on hard enough; then an ideal resistor of the target value stands in, so
            the power can still be computed
  ADC       the ladder spans 0 to the read voltage; codes must be 0 to 7 (no device
            mismatch is simulated, so a pass here is necessary, not sufficient)
  share     header + ADC as a share of the macro, per rows on, all column current
            counted as the header's. WL, decoder and mux selects from rram_macro_power.json

Usage (from the repo root, after rram_macro_power.py):
  python3 targets/rram-sram-ensemble/spice/readout/sweep_read_voltage.py
"""
import json
import os

from characterize_readout import (
    ASSUMED, COLUMN_MUX, DESIGN, TARGET, TCK, convert, header_mos, mean_cell_conductance, read_path, size_header,
)
from thick_model import access_on_ohm

READ_VOLTAGES = (0.1, 0.2, 0.3, 0.45, 0.6, 0.9, 1.2)
ROWS_SHOWN = ("8", "16", "32", "64")


def main() -> None:
    results = os.path.join(TARGET, "default", "results")
    macro = json.load(open(os.path.join(results, "rram_macro_power.json")))["by_rows_on"]
    cell = DESIGN["rram_macro"]["cell"]
    r_min = min(cell["resistance_kohm_approx_fig8"]) * 1e3
    r_target = ASSUMED["header_over_min_cell_resistance"] * r_min
    g_mean = mean_cell_conductance(access_on_ohm())
    readouts = DESIGN["rram_macro"]["readout"]["readout_circuits"]
    mac_s = COLUMN_MUX * TCK

    points = {}
    for read_v in READ_VOLTAGES:
        width = size_header(r_target, r_min, read_v)
        header = header_mos(width) if width is not None else f"RH top vrd {r_target}"
        lsb = read_v / 8
        codes, energy = convert([(k + 0.5) * lsb for k in range(8)], read_v)
        adc_mW = readouts * energy / TCK * 1e3
        rows = {}
        for n in ROWS_SHOWN:
            p = read_path(header, 1 / (int(n) * g_mean), read_v)
            column = readouts * p["read_supply_J"] / mac_s * 1e3
            m = macro[n]
            others = m["wl_mW"] + m["decoder_mW"] + readouts * p["mux_select_J"] / mac_s * 1e3 \
                + m["programming_mux_leakage_mW"]
            total = column + adc_mW + others
            rows[n] = {"read_node_V": p["top_V"], "column_mW": column, "others_mW": others, "total_mW": total,
                       "share_pct": 100 * (column + adc_mW) / total}
        points[str(read_v)] = {"header_width_nm": width, "adc_codes": codes, "adc_correct": codes == list(range(8)),
                               "adc_mW": adc_mW, "by_rows_on": rows}

    path = os.path.join(results, "read_voltage_sweep.json")
    with open(path, "w") as f:
        json.dump({"target_header_ohm": r_target, "points": points}, f, indent=2)

    print(f"{'read V':>6} {'header W (nm)':>14} {'ADC ok':>7} {'ADC mW':>7} | share % at rows on "
          + " ".join(f"{n:>6}" for n in ROWS_SHOWN) + " | read node V at the same rows")
    for v, p in points.items():
        width = "none" if p["header_width_nm"] is None else f"{p['header_width_nm']:.0f}"
        r = p["by_rows_on"]
        print(f"{v:>6} {width:>14} {str(p['adc_correct']):>7} {p['adc_mW']:7.3f} |                     "
              + " ".join(f"{r[n]['share_pct']:6.1f}" for n in ROWS_SHOWN) + " | "
              + " ".join(f"{r[n]['read_node_V']:.3f}" for n in ROWS_SHOWN))
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
