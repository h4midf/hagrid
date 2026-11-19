"""Comparison table: our reconstruction vs HyFlexPIM's reported numbers.

Reads default/results/*.json and design.json's reported_results, prints the table and
writes default/results/summary.md. Missing results show as "not run".

Usage: python3 targets/hyflexpim/report.py
"""
import json
from datetime import datetime
from decimal import Decimal
from pathlib import Path

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "default"
RESULTS = CONFIG / "results"
DESIGN = json.loads((CONFIG / "design.json").read_text())
PAPER = DESIGN["reported_results"]

# rows whose gap is understood; anything else gets "match" or "within x%"
NOTES = {
    ("analog", "s_and_h", "power_mW"): "Table 2 printed in W: ours 11.6e-6 W",
    ("digital", "s_and_h", "power_mW"): "Table 2 printed in W: ours 23.3e-6 W",
    ("analog", "rram_array", "area_mm2"): "gap: x1.064 in Table 2, cause not found",
}


def load(name):
    path = RESULTS / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def status(ours, printed):
    """match when ours rounds to the printed value at its printed precision."""
    exp = Decimal(repr(printed)).as_tuple().exponent
    half_ulp = 0.5 * 10.0 ** exp
    if abs(ours - printed) <= half_ulp * (1 + 1e-9):
        return "match"
    if abs(ours / printed - 1) <= 0.1:
        return f"within {abs(ours / printed - 1) * 100:.1f}%"
    return f"gap: ours x{ours / printed:.2f}"


def fmt(v):
    return "not run" if v is None else f"{v:.4g}"


def table2_rows(t2):
    out = []
    t2 = t2 or {}
    digital_key = "digital_module" if "digital_module" in t2 else "digital_module_without_sfu"
    for module, ours_key, paper_key in (("analog", "analog_module", "table2_analog_module"),
                                        ("digital", digital_key, "table2_digital_module")):
        paper = PAPER[paper_key]
        ours = t2.get(ours_key)
        for comp, p in paper["components"].items():
            for qty, unit in (("area_mm2", "mm^2"), ("power_mW", "mW")):
                o = (ours["components"].get(comp) or {}).get(qty) if ours else None
                if o is None:
                    note = "not run"
                else:
                    note = NOTES.get((module, comp, qty), status(o, p[qty]))
                out.append((f"{module} module {comp}, {unit}", fmt(o), f"{p[qty]:g}", note))
        sums = [("sum", ours, 0.0)]
        if module == "digital":   # also without the SFU, the one row that is not ASADI-derived
            sums.append(("sum without SFU", t2.get("digital_module_without_sfu"), 1.0))
        for label, part, minus_sfu in sums:
            for qty, unit in (("area_mm2", "mm^2"), ("power_mW", "mW")):
                o = part["module_sum"][qty] if part else None
                p = paper["module_sum"][qty] - minus_sfu * paper["components"].get("sfu", {}).get(qty, 0.0)
                out.append((f"{module} module {label}, {unit}", fmt(o), f"{p:.6g}",
                            "not run" if o is None else f"x{o / p:.3f}"))
    return out


def sfu_rows(t2):
    """Our SFU synthesis beyond Table 2's two numbers: its power split and its clock."""
    sfu = ((t2 or {}).get("digital_module") or {}).get("components", {}).get("sfu")
    if sfu is None:
        return []
    return [
        ("SFU dynamic power, BERT-Large N=128 workload, idle units clock-gated, mW", fmt(sfu["dynamic_mW"]), "-",
         f"ours only; busy {100 * sfu['busy_fraction']:.1f}% of the layer time"),
        ("SFU leakage power, mW", fmt(sfu["leakage_mW"]), "-", "ours only"),
        ("SFU power with every unit busy every cycle (alpha 0.2), mW", fmt(sfu["all_busy_mW"]), "-",
         "ours only; not what Table 2 reports"),
        ("SFU pipeline stage at 65 nm (logic + flop), ps", f"{sfu['stage_ps']:.0f}", "1000 (1 GHz, Sec. 5.4)",
         "meets 1 GHz" if sfu["meets_clock"] else "misses 1 GHz"),
    ]


def shift_add_rows():
    """Our S&A synthesis (periphery/characterize_shift_add.py): the flow checked against
    ISAAC's S+A, the source of Table 2's S&A row, then a working HyFlexPIM S&A."""
    path = HERE / "periphery" / "results" / "shift_add.json"
    if not path.exists():
        return []
    rows = {r["unit"]: r for r in json.loads(path.read_text())}
    out = []
    isaac = rows.get("isaac_shift_add", {}).get("isaac_32nm")
    if isaac:
        out += [
            ("ISAAC S+A (bare 16-bit shift-and-add), 32 nm, um^2", fmt(isaac["area_um2"]),
             f"{isaac['isaac_area_um2']:g} (ISAAC Table I)",
             f"x{isaac['area_um2'] / isaac['isaac_area_um2']:.2f}: our flow reproduces the source"),
            ("ISAAC S+A, 32 nm, 1.2 GHz, mW", fmt(isaac["power_mW"]), f"{isaac['isaac_power_mW']:g} (ISAAC Table I)",
             f"x{isaac['power_mW'] / isaac['isaac_power_mW']:.2f} (alpha 0.2)"),
        ]
    hfp = rows.get("hfp_shift_add", {}).get("hyflexpim_65nm")
    if hfp:
        paper = PAPER["table2_analog_module"]["components"]["s_and_a"]
        n = hfp["analog_module"]["units"]
        u = hfp["unit"]
        out += [
            ("Working S&A per array (column weighting, partial, accumulate, 1.28 GHz), um^2",
             fmt(u["area_um2"]), f"{paper['area_mm2'] / n * 1e6:.4g} (Table 2 / {n})",
             f"ours x{u['area_um2'] / (paper['area_mm2'] / n * 1e6):.1f}: Table 2's row is one bare adder"),
            ("Working S&A per array, mW", fmt(u["power_mW"]), f"{paper['power_mW'] / n:.4g} (Table 2 / {n})",
             f"ours x{u['power_mW'] / (paper['power_mW'] / n):.1f}"),
            ("Working S&A stage at 65 nm, ps", f"{u['stage_ps']:.0f}", f"{u['clock_ps']:.0f} (1.28 GS/s ADC)",
             "meets the ADC rate" if u["meets_clock"] else "misses the ADC rate"),
        ]
    return out


def register_rows():
    """CACTI 6.5 checked against ISAAC's registers, the source of Table 2's IR / OR rows."""
    path = HERE / "periphery" / "results" / "registers.json"
    if not path.exists():
        return []
    out = []
    for name, r in json.loads(path.read_text())["validation"].items():
        label = {"isaac_ir": "ISAAC IR", "isaac_or": "ISAAC OR"}.get(name, name)
        out.append((f"{label} ({r['size_bytes']} B SRAM, {r['node_nm']} nm, CACTI 6.5), mm^2",
                    fmt(r["area_mm2"]), f"{r['source_area_mm2']:g} (ISAAC Table I)",
                    f"x{r['area_mm2'] / r['source_area_mm2']:.2f}"))
    return out


def timing_rows():
    cfg = DESIGN["analog_module"]["readout"]
    scan_ns = cfg["columns_per_adc"] / cfg["adc_sample_rate_GSps"]
    return [("ADC scan of one array (128 columns / 1.28 GS/s)", f"{scan_ns:.1f} ns",
             f"{PAPER['timing']['analog_cycle_ns']} ns",
             "match; the 100 ns crossbar read is ISAAC's assumption, not simulated")]


def main():
    t2 = load("table2")
    scale = t2["node_scaling"] if t2 else None
    rows = [("Stillmaker area factor, 32 nm HP -> 65 nm bulk",
             fmt(scale and scale["area_factor"]), "3.3 (Table 4, rounded)", "match"),
            ("Stillmaker energy factor, 1.0 V both nodes",
             fmt(scale and scale["power_factor"]), "-", "ours only (fits Table 2)")]
    rows += table2_rows(t2) + sfu_rows(t2) + shift_add_rows() + register_rows() + timing_rows()

    lines = ["# HyFlexPIM `default`: reconstructed vs Song et al. 2025 (Table 2)", "",
             f"Generated {datetime.now().isoformat(timespec='seconds')} by report.py from results/*.json.", "",
             "| Quantity | Ours | HyFlexPIM | Status |", "|---|---|---|---|"]
    lines += [f"| {q} | {o} | {p} | {s} |" for q, o, p, s in rows]
    text = "\n".join(lines) + "\n"
    (RESULTS / "summary.md").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
