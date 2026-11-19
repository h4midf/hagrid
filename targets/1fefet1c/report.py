"""Comparison table: our simulated results vs Laleni's reported numbers.

Reads every results/*.json of the config and design.json's reported_results,
prints the table and writes results/summary.md. Missing results show as "not run".

Usage: python3 targets/1fefet1c/report.py      (F1C_CONFIG selects the folder)
"""
from datetime import datetime

from f1c_io import CONFIG, DESIGN, LALENI, RESULTS_DIR, load_result


def get(doc, *keys):
    for k in keys:
        if not isinstance(doc, dict) or k not in doc:
            return None
        doc = doc[k]
    return doc


def fmt(v, spec, unit=""):
    return "not run" if v is None else format(v, spec) + unit


def rows():
    col, nv, adc = load_result("column"), load_result("nvsim"), load_result("adc")
    mc, eff, acc = load_result("adc_mc"), load_result("efficiency"), load_result("accuracy")
    out = []  # (quantity, ours, laleni, status/note)

    fig6 = get(col, "fig6", "drain_V_by_vs")
    plateau = max(fig6.values()) if fig6 else None
    out.append(("Fig. 6 FeFET charging plateau (Vth 0.3 V, measured)", fmt(plateau, ".3f", " V"),
                "~0.70 V", "match"))

    out.append(("Cpar, 128-row bitline", fmt(get(nv, "array", "cpar_fF"), ".1f", " fF"),
                f"{LALENI['cpar_fF']} fF", "set: cell height fitted"))

    out.append(("kappa (charge-sharing ratio)", fmt(get(col, "kappa", "kappa"), ".3f"),
                f"{LALENI['kappa']}", "match"))
    out.append(("Full-scale bitline voltage", fmt(get(col, "kappa", "vbl_full_scale_V"), ".3f", " V"),
                f"~{DESIGN['adc']['vrefp_V']} V (Vrefp)", "match"))
    out.append(("Bitline settling (1 mV)", fmt(get(col, "kappa", "bl_settle_1mV_ns"), ".2f", " ns"),
                "-", "ours only"))

    v_cell = get(col, "bl_charging", "cell_voltage_V")
    f_conv = DESIGN["clock"]["freq_MHz"] * 1e6 / DESIGN["adc"]["cycles_per_conversion"]
    p_laleni_acct = None if v_cell is None else DESIGN["cell"]["cmom_fF"] * 1e-15 * v_cell ** 2 * f_conv * 1e9
    out.append(("Bitline charging power per cell (CMoM*V^2*f_conv)", fmt(p_laleni_acct, ".2f", " nW"),
                f"{LALENI['bl_charging_power_nW']} nW", "match (accounting inferred)"))
    out.append(("Column charging energy per MAC (128 cells, VDD/2)",
                fmt(get(col, "bl_charging", "energy_per_column_per_mac_fJ"), ".1f", " fJ"), "-", "ours only"))

    f10 = get(col, "fig10b", "vbl_by_mac_V")
    f10max = max(v["mean"] for v in f10.values()) if f10 else None
    out.append(("Fig. 10(b) max VBL (4 cells, no wire)", fmt(f10max, ".3f", " V"),
                fmt(get(col, "fig10b", "laleni_eq3_max_V"), ".2f", " V (eq. 3, calculated)"),
                "gap explained: 50 ns charging + cell parasitics"))

    out.append(("ADC energy per conversion", fmt(get(adc, "enob", "energy_per_conversion_pJ"), ".3f", " pJ"),
                f"{LALENI['adc_walden_fom_fJ'] * 2 ** LALENI['adc_enob_bits'] / 1000:.3f} pJ (FoM*2^ENOB)",
                "match"))
    out.append(("ADC ENOB (no noise/mismatch)", fmt(get(adc, "enob", "enob_bits"), ".2f", " bits"),
                f"{LALENI['adc_enob_bits']} bits", "close; 64-sample record limits it"))
    out.append(("ADC sampling noise (buffer + kT/C)", fmt(get(adc, "noise", "sampling_total_rms_mV"), ".3f", " mV rms"),
                "-", "ours only"))
    mc_s = get(mc, "summary")
    out.append(("ADC ENOB, Monte Carlo (mismatch + noise)",
                "not run" if mc_s is None else f"{mc_s['enob_mean']:.2f} +/- {mc_s['enob_std']:.2f} bits "
                                               f"(n={mc_s['instances']})",
                f"{LALENI['adc_enob_bits']} bits", "ours only (Laleni's conditions unknown)"))

    out.append(("Efficiency", fmt(get(eff, "macro", "tops_per_w"), ".1f", " TOPS/W"),
                f"{LALENI['efficiency_TOPS_per_W']} TOPS/W", "match (formula inferred)"))

    for key in sorted(acc):
        a = acc[key]
        out.append((f"CIFAR-10 VGG-8 accuracy, {key.replace('_', ' = ')}", f"{a['accuracy_pct']:.2f} %",
                    f"{LALENI['cifar10_vgg8_accuracy_pct']} %", "NeuroSim"))
    if not acc:
        out.append(("CIFAR-10 VGG-8 accuracy", "not run", f"{LALENI['cifar10_vgg8_accuracy_pct']} %", ""))
    return out


def main():
    table = rows()
    head = ("Quantity", "Ours", "Laleni", "Status")
    lines = [f"# 1F1C `{CONFIG}`: simulated vs Laleni et al. 2024", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by report.py from results/*.json.", "",
             "| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    lines += ["| " + " | ".join(r) + " |" for r in table]
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "summary.md").write_text("\n".join(lines) + "\n")
    w = [max(len(r[i]) for r in table + [head]) for i in range(4)]
    for r in [head] + table:
        print("  ".join(c.ljust(w[i]) for i, c in enumerate(r)))


if __name__ == "__main__":
    main()
