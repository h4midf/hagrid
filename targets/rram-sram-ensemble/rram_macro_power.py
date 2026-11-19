"""Power of the RRAM macro at the blocks Sec. V-F lists (crossbar, WL driver and shifter,
decoder, flash ADCs, pMOS headers, BL / SL / column muxes), and the share of the header and
ADC, to compare with the paper's 90%. Reads default/results/{readout,wl}.json and
periphery/results/{rram_decoder,rram_shift_add}.json; writes
default/results/rram_macro_power.json.

The share is given two ways, since the paper does not say how it split the supply current
between the header and the crossbar it feeds:
  supply view   all the column current drawn through the header counts as the header's
  header only   only the power dissipated in the header; the rest is the crossbar's

WL activity: input bits are independent with probability rows_on / rows of being 1 and a
WL holds its value over one MAC, so a WL rises with probability a (1 - a) per MAC.
The BL / SL programming muxes are idle during compute: only their leakage counts.
Shift-and-add is not in Sec. V-F's list: it is reported, but left out of the share.

Usage (from the repo root): python3 targets/rram-sram-ensemble/rram_macro_power.py
"""
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "default" / "results"
DESIGN = json.loads((HERE / "default" / "design.json").read_text())
PAPER_SHARE_PCT = DESIGN["reported_results"]["rram_macro_power_share_header_and_adc_pct"]


def load(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"{path}: run its characterization first")
    return json.loads(path.read_text())


def main() -> None:
    readout = load(RESULTS / "readout.json")
    wl = load(RESULTS / "wl.json")
    decoder = load(HERE / "periphery" / "results" / "rram_decoder.json")[0]["chip_65nm"]
    shift_add = load(HERE / "periphery" / "results" / "rram_shift_add.json")[0]["chip_65nm"]

    rows = DESIGN["rram_macro"]["array"]["rows"]
    mac_ns = DESIGN["rram_macro"]["readout"]["cycles_per_mac"] * readout["clock_ns"]
    adc = readout["flash_adc"]["all_readouts_power_mW"]
    pair_pJ = wl["energy_pJ_per_toggle_pair"]["total"]

    points = {}
    for n, c in readout["by_rows_on"].items():
        a = int(n) / rows
        wl_mW = rows * a * (1 - a) * pair_pJ / mac_ns
        others = wl_mW + decoder["power_mW"] + c["all_readouts_mux_select_mW"] + c["programming_mux_leakage_mW"]
        supply, header = c["all_readouts_supply_mW"], c["all_readouts_header_only_mW"]
        total = supply + adc + others
        points[n] = {"adc_mW": adc, "column_supply_mW": supply, "header_only_mW": header,
                     "crossbar_and_mux_mW": supply - header, "wl_mW": wl_mW, "decoder_mW": decoder["power_mW"],
                     "mux_select_mW": c["all_readouts_mux_select_mW"],
                     "programming_mux_leakage_mW": c["programming_mux_leakage_mW"], "total_mW": total,
                     "share_supply_view_pct": 100 * (supply + adc) / total,
                     "share_header_only_pct": 100 * (header + adc) / total}

    out = {"paper_share_pct": PAPER_SHARE_PCT, "shift_add_mW_not_in_share": shift_add["power_mW"], "by_rows_on": points}
    path = RESULTS / "rram_macro_power.json"
    path.write_text(json.dumps(out, indent=2) + "\n")
    print(f"{'rows':>4} {'ADC':>7} {'column':>8} {'header':>8} {'WL':>7} {'decoder':>8} {'mux sel':>8} "
          f"{'total':>8} {'supply%':>8} {'hdr-only%':>9}")
    for n, p in points.items():
        print(f"{n:>4} {p['adc_mW']:7.3f} {p['column_supply_mW']:8.3f} {p['header_only_mW']:8.3f} {p['wl_mW']:7.3f} "
              f"{p['decoder_mW']:8.4f} {p['mux_select_mW']:8.4f} {p['total_mW']:8.3f} "
              f"{p['share_supply_view_pct']:8.1f} {p['share_header_only_pct']:9.1f}")
    print(f"paper: {PAPER_SHARE_PCT}%; shift-and-add {shift_add['power_mW']:.4f} mW (not in the share); wrote {path}")


if __name__ == "__main__":
    main()
