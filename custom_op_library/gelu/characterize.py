"""Synthesises the gelu circuits (gelu.v, gelu_tanh.v, ibert_gelu.v, wang_tcad_gelu24.v,
gqa_lut_gelu8.v) at both widths and writes results/gelu.json. The tables are fixed and
hand-written."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "gelu.v")
IBERT_RTL = os.path.join(HERE, "rtl", "ibert_gelu.v")
WANG_RTL = os.path.join(HERE, "rtl", "wang_tcad_gelu24.v")
GQA_RTL = os.path.join(HERE, "rtl", "gqa_lut_gelu8.v")
TANH_RTL = os.path.join(HERE, "rtl", "gelu_tanh.v")
CASES = [
    (32, 20, "Q12.20", "mean abs err 2.5e-5, max abs err 3.9e-4 vs true GELU; 128 segments tried first, "
                       "missed the bar"),
    (16, 8, "Q8.8", "mean abs err 1.0e-3, max abs err 8.7e-3 vs true GELU"),
]
TANH_CASES = [
    (32, 20, "Q12.20", "mean abs err 2.5e-5, max abs err 3.9e-4 vs true tanh-form GELU"),
    (16, 8, "Q8.8", "mean abs err 1.0e-3, max abs err 8.7e-3 vs true tanh-form GELU"),
]


def gelu_tanh_rows():
    """gelu_tanh.v (the tanh form, GPT-2's gelu_new) at both widths: gelu.v's circuit
    with a tanh-form table."""
    rows = []
    for w, frac, fmt, err_note in TANH_CASES:
        res = synth.synth([TANH_RTL], "gelu_tanh", {"W": w, "FRAC": frac})
        rows.append(report.row("gelu_tanh", {"W": w, "segments": 256, "domain": "+-8", "in": fmt, "out": fmt},
                               "256-segment piecewise-linear (exact tanh-form GELU breakpoints)", res,
                               f"validated: {err_note} (see test_gelu_tanh.py)"))
        print(f"  gelu_tanh W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    return rows


def main():
    rows = gelu_tanh_rows()
    for w, frac, fmt, err_note in CASES:
        res = synth.synth([RTL], "gelu", {"W": w, "FRAC": frac})
        row = report.row("gelu", {"W": w, "segments": 256, "domain": "+-8", "in": fmt, "out": fmt},
                          "256-segment piecewise-linear (exact GELU breakpoints)", res,
                          f"validated: {err_note} (see approximation_error.py)")
        rows.append(row)
        print(f"  gelu W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    # ibert_gelu at W=16 only: at W=32, ABC's mapping of its chained multiplies does not
    # finish (14+ min); the circuit is bit-exact at both widths (test_ibert_gelu.py)
    res16 = synth.synth([IBERT_RTL], "ibert_gelu", {"W": 16, "FRAC": 8})
    rows.append(report.row("ibert_gelu", {"W": 16, "in": "Q8.8", "out": "Q8.8"},
                            "I-BERT (ICML 2021) clipped-quadratic erf fit", res16,
                            "verified clean on bert-large-uncased: 7.1713 ppl vs 7.3563 baseline, 62.17% top-1 "
                            "(see paper/approximate/selected_approximate.md idea 5). W=32/Q12.20 is bit-exact "
                            "correct (test_ibert_gelu.py) but did NOT synthesize within 14+ minutes through "
                            "this project's ABC pipeline -- not characterized, disclosed limitation."))
    print(f"  ibert_gelu W=16: area {res16['area_um2']:.0f} um^2, delay {res16['delay_ps']} ps", flush=True)
    print("  ibert_gelu W=32: SKIPPED -- did not synthesize within 14+ minutes (see note in results)", flush=True)

    # Wang-TCAD (a step function, slope 0) and GQA-LUT (piecewise linear): hardwired
    # constants, as softmax_row8.v's exp table; a width that fails to synthesize is
    # reported and skipped
    for rtl, name, segs, method, note in (
        (WANG_RTL, "wang_tcad_gelu24", 24, "24-entry nonuniform step-function LUT (Wang et al. TCAD'25)",
         "bert-large-calibrated breakpoints/outputs; see paper/approximate/selected_approximate.md idea 4"),
        (GQA_RTL, "gqa_lut_gelu8", 8, "8-entry genetic piecewise-linear LUT (GQA-LUT, DAC'24)",
         "verified clean on bert-large-uncased: 7.2420 ppl / 61.18% top-1 vs 7.3563/61.18% baseline; "
         "see paper/approximate/selected_approximate.md idea 7"),
    ):
        for w, frac, fmt in ((32, 20, "Q12.20"), (16, 8, "Q8.8")):
            try:
                res = synth.synth([rtl], name, {"W": w, "FRAC": frac})
                rows.append(report.row(name, {"W": w, "segments": segs, "in": fmt, "out": fmt}, method, res, note))
                print(f"  {name} W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
            except Exception as exc:
                print(f"  {name} W={w}: SYNTHESIS FAILED/SKIPPED ({exc}) -- functional correctness still "
                      f"verified via test_{name}.py", flush=True)
    report.save(HERE, "gelu", rows)


if __name__ == "__main__":
    main()
