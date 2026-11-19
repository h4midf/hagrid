"""Comparison table: our reconstruction vs Krishnan et al. (TCAD 2022) reported numbers.

Reads default/results/{rram_macro_power,sram_over_rram,read_voltage_sweep}.json and
design.json's reported_results, prints the table and writes default/results/summary.md.
Missing results show as "not run". A row is "match" within MATCH_PCT of the paper.

The SRAM-over-RRAM ratios come from fixed macros, so one single-macro ratio serves every
network; the paper's numbers imply a different one per network. The "best any fixed macro"
rows give the smallest worst-case error any single ratio could reach.

Usage (from the repo root): python3 targets/rram-sram-ensemble/report.py
"""
import json
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "default" / "results"
DESIGN = json.loads((HERE / "default" / "design.json").read_text())
PAPER = DESIGN["reported_results"]
MATCH_PCT = 15
ROWS_ON = "16"   # rows on per MAC for the power rows: about the 16% input-bit density measured on real data


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def status(ours, paper):
    off = abs(ours / paper - 1) * 100
    return f"match ({off:.0f}% off)" if off <= MATCH_PCT else f"gap: ours x{ours / paper:.2f}"


def best_fixed(implied: list) -> str:
    lo, hi = min(implied), max(implied)
    return f"{2 * lo * hi / (lo + hi):.4f} at best, {100 * (hi - lo) / (hi + lo):.0f}% worst-case error"


def timing_rows():
    rm = DESIGN["rram_macro"]["readout"]
    cycle_ns = 1e3 / DESIGN["clock"]["freq_MHz"]
    return [("RRAM MAC per input bit (8-to-1 column mux at 100 MHz)", f"{rm['cycles_per_mac']} cycles, "
             f"{rm['cycles_per_mac'] * cycle_ns:.0f} ns", "8 cycles (Sec. III-D-1)", "match (paper's own value)")]


def share_rows(power):
    paper = PAPER["rram_macro_power_share_header_and_adc_pct"]
    if power is None:
        return [("Header + ADC share of RRAM macro power, %", "not run", f"{paper}", "not run")]
    out = []
    for n, p in power["by_rows_on"].items():
        out.append((f"Header + ADC share of RRAM macro power, {n} rows on (all column current as the header's), %",
                    f"{p['share_supply_view_pct']:.1f}", f"{paper} (Sec. V-F)",
                    status(p["share_supply_view_pct"], paper)))
    return out


def ratio_rows(ratios):
    if ratios is None:
        return [("SRAM over RRAM area and power", "not run", "Fig. 12", "not run")]
    out = []
    for name, v in ratios["networks"].items():
        a = v["area_pct"]
        out.append((f"SRAM over RRAM area, {name}, %", f"{a['ours']:.2f} (Fig. 6 layout: {a['from_fig6_layout']:.2f})",
                    f"{a['paper']} (Fig. 12a)", status(a["ours"], a["paper"])))
    for name, v in ratios["networks"].items():
        p = v["power_pct"]
        ours = p["ours_by_rows_on"][ROWS_ON]
        spread = "/".join(f"{x:.2f}" for x in p["ours_by_rows_on"].values())
        out.append((f"SRAM over RRAM power, {name}, {ROWS_ON} rows on, %",
                    f"{ours:.2f} (8/16/32 rows: {spread})", f"{p['paper']} (Fig. 12b)", status(ours, p["paper"])))
    one = ratios["one_macro"]
    nets = ratios["networks"].values()
    out += [
        ("Single-macro area ratio (SRAM / RRAM)", f"{one['area_ratio']:.4f} (Fig. 6 layout: "
         f"{one['area_ratio_from_fig6_layout']:.4f})",
         " / ".join(f"{v['implied_one_macro_area_ratio']:.4f}" for v in nets) + " (implied per network)",
         "best any fixed macro: " + best_fixed([v["implied_one_macro_area_ratio"] for v in nets])),
        (f"Single-macro power ratio (SRAM / RRAM), {ROWS_ON} rows on",
         f"{one['power_ratio_by_rows_on'][ROWS_ON]:.4f}",
         " / ".join(f"{v['implied_one_macro_power_ratio']:.4f}" for v in nets) + " (implied per network)",
         "best any fixed macro: " + best_fixed([v["implied_one_macro_power_ratio"] for v in nets])),
        ("SRAM macro (logic + array), um^2 / mW", f"{ratios['sram_macro']['area_um2']:.0f} / "
         f"{ratios['sram_macro']['power_mW']:.4f}", "-", "ours only"),
        (f"RRAM macro, um^2 / mW at {ROWS_ON} rows on", f"{ratios['rram_macro']['area_um2']:.0f} / "
         f"{ratios['rram_macro']['power_mW_by_rows_on'][ROWS_ON]:.3f}", "-", "ours only"),
    ]
    return out


def read_voltage_rows(sweep):
    if sweep is None:
        return []
    out = []
    for v, p in sweep["points"].items():
        header = "no pMOS header reaches it" if p["header_width_nm"] is None else f"header {p['header_width_nm']:.0f} nm"
        out.append((f"Header + ADC share at a {v} V read supply, {ROWS_ON} rows on, %",
                    f"{p['by_rows_on'][ROWS_ON]['share_pct']:.1f}", "-", f"ours only; {header}"))
    return out


def accuracy_rows():
    return [(f"Accuracy, {name}, %", "not run", f"{r['accuracy_pct']} (Table III)", "not run")
            for name, r in PAPER["accuracy_table3"].items()]


def main():
    rows = (timing_rows() + share_rows(load("rram_macro_power")) + ratio_rows(load("sram_over_rram"))
            + read_voltage_rows(load("read_voltage_sweep")) + accuracy_rows())
    lines = ["# RRAM/SRAM ensemble `default`: reconstructed vs Krishnan et al. 2022", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by report.py from results/*.json. "
             f"'match' = within {MATCH_PCT}% of the paper.", "",
             "| Quantity | Ours | Paper | Status |", "|---|---|---|---|"]
    lines += [f"| {q} | {o} | {p} | {s} |" for q, o, p, s in rows]
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
