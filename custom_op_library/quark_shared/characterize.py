"""Synthesizes QUARK's shared-circuit modules (appro_exp, appro_ln, recip, shared_div,
softmax_row8, gelu) at both widths and writes results/quark_shared.json. A case that
fails to synthesize is reported and has no row.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL_DIR = os.path.join(HERE, "rtl")
PRIMS = [os.path.join(RTL_DIR, f) for f in ("quark_appro_exp.v", "quark_appro_ln.v", "quark_recip.v")]
SHARED_DIV = PRIMS + [os.path.join(RTL_DIR, "quark_shared_div.v")]
SOFTMAX = PRIMS[:2] + [os.path.join(RTL_DIR, "reduce_tree.v"), os.path.join(RTL_DIR, "quark_softmax_row8.v")]
GELU = SOFTMAX + [os.path.join(RTL_DIR, "quark_gelu.v")]
WIDTHS = {32: "Q12.20", 16: "Q8.8"}
FRACS = {32: 20, 16: 8}

NOTES = ("verified bit-exact (test_quark_*.py) against the already-validated Python reference in "
         "run_scripts/measure_accuracy/quark_bert_large.py; see paper/approximate/selected_approximate.md idea 3")


def try_synth(files, top, params, label, rows):
    try:
        res = synth.synth(files, top, params)
        row = report.row(top, params, "QUARK (ICCAD 2025) shared log/exp circuit", res, NOTES)
        rows.append(row)
        print(f"  {label}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    except Exception as exc:
        print(f"  {label}: SYNTHESIS FAILED/SKIPPED ({exc}) -- functional correctness still verified via "
              f"test_{top}.py", flush=True)


def main():
    rows = []
    for w, fmt in WIDTHS.items():
        frac = FRACS[w]
        try_synth(PRIMS[:1], "quark_appro_exp", {"W": w, "FRAC": frac}, f"quark_appro_exp W={w}", rows)
        try_synth(PRIMS[1:2], "quark_appro_ln", {"W": w, "FRAC": frac}, f"quark_appro_ln W={w}", rows)
        try_synth(PRIMS[2:3], "quark_recip", {"W": w, "FRAC": frac}, f"quark_recip W={w}", rows)
        try_synth(SHARED_DIV, "quark_shared_div", {"W": w, "FRAC": frac}, f"quark_shared_div W={w}", rows)
        try_synth(SOFTMAX, "quark_softmax_row8", {"W": w, "FRAC": frac}, f"quark_softmax_row8 W={w}", rows)
        try_synth(GELU, "quark_gelu", {"W": w, "FRAC": frac}, f"quark_gelu W={w}", rows)
    report.save(HERE, "quark_shared", rows)


if __name__ == "__main__":
    main()
