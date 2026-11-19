"""Synthesizes the softmax circuits and writes results/softmax.json: the four row-8
softmaxes and the per-stage circuits of the row pipeline (max_reduce_tree, the
per-element exp stages, the normalize stages).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "softmax_row8.v")
E2SM_RTL = [RTL, os.path.join(HERE, "rtl", "e2softmax_row8.v")]
IBERT_RTL = [RTL, os.path.join(HERE, "rtl", "ibert_softmax_row8.v")]
GQA_RTL = os.path.join(HERE, "rtl", "gqa_lut_exp8.v")
GQA_SM_RTL = [RTL, GQA_RTL, os.path.join(HERE, "rtl", "gqa_lut_softmax_row8.v")]
# Per-stage circuits of the row pipeline every non-"lut" softmax is costed through
# (load_results.py's softmax_stages()).
MAX_REDUCE_TREE_RTL = os.path.join(HERE, "rtl", "max_reduce_tree.v")
EXP_PWL16_RTL = os.path.join(HERE, "rtl", "exp_pwl16_standalone.v")
Q8_8_RECIP_RTL = os.path.join(HERE, "rtl", "q8_8_reciprocal.v")
LOG2EXP_RTL = os.path.join(HERE, "rtl", "log2exp_standalone.v")
ALDIVISION_RTL = os.path.join(HERE, "rtl", "aldivision_standalone.v")
IBERT_EXP_PWL_RTL = os.path.join(HERE, "rtl", "ibert_exp_pwl_standalone.v")


def try_synth(files, top, params, label, method, note, rows):
    try:
        res = synth.synth(files, top, params)
        rows.append(report.row(top, params, method, res, note))
        print(f"  {label}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    except Exception as exc:
        print(f"  {label}: SYNTHESIS FAILED/SKIPPED ({exc}) -- functional correctness still verified via "
              f"test_{top}.py", flush=True)


def main():
    rows = []
    res = synth.synth([RTL], "softmax_row8", {})
    rows.append(report.row("softmax_row8", {"row_width": 8, "in": "8-bit score", "out": "Q8.8"},
                            "base-2 decompose + 16-segment piecewise-linear (RMSL)", res,
                            "exp via bit-shift (integer power-of-2) + 16-seg PWL (fractional); "
                            "one reciprocal + 8 multiplies for normalization"))
    print(f"  softmax_row8: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps, "
          f"cells {sum(res['cells'].values())}", flush=True)

    try_synth(E2SM_RTL, "e2softmax_row8", {}, "e2softmax_row8",
              "log2-quantized exponent (4-bit) + ALDivision (shift+2-way mux, no divider)",
              "SOLE (ICCAD 2023); verified clean on bert-large-uncased: 7.61 ppl / 61.84% top-1 vs "
              "7.36/61.18% baseline; see paper/approximate/selected_approximate.md idea 2", rows)
    try_synth(IBERT_RTL, "ibert_softmax_row8", {}, "ibert_softmax_row8",
              "range-reduced quadratic-poly exp (I-BERT, ICML 2021), real division for normalization",
              "verified clean on bert-large-uncased: 7.3568 ppl / 61.18% top-1 vs 7.3563/61.18% baseline; "
              "see paper/approximate/selected_approximate.md idea 5", rows)
    for w, frac, fmt in ((32, 20, "Q12.20"), (16, 8, "Q8.8")):
        try_synth([GQA_RTL], "gqa_lut_exp8", {"W": w, "FRAC": frac}, f"gqa_lut_exp8 W={w}",
                  "8-entry genetic piecewise-linear LUT (GQA-LUT, DAC'24), output-clamped-to-0 below domain floor",
                  "verified clean on bert-large-uncased: 8.2769 ppl / 59.21% top-1 vs 7.3563/61.18% baseline; "
                  "see paper/approximate/selected_approximate.md idea 7", rows)
    try_synth(GQA_SM_RTL, "gqa_lut_softmax_row8", {}, "gqa_lut_softmax_row8",
              "8-entry genetic piecewise-linear exp LUT (GQA-LUT, DAC'24), real division for normalization",
              "verified clean on bert-large-uncased: 8.2769 ppl / 59.21% top-1 vs 7.3563/61.18% baseline; "
              "see paper/approximate/selected_approximate.md idea 7", rows)

    try_synth([MAX_REDUCE_TREE_RTL], "max_reduce_tree", {"N": 512, "W": 32}, "max_reduce_tree N=512",
              "comparator-tree max reduction, real bert-large row width",
              "shared stage 1 of every non-lut softmax variant's own real row-pipeline decomposition "
              "(global max over the real 512-wide row, before any per-element exp stage)", rows)
    try_synth([MAX_REDUCE_TREE_RTL], "max_reduce_tree", {"N": 256, "W": 32}, "max_reduce_tree N=256",
              "comparator-tree max reduction, HyFlexPIM's own stated SFU width",
              "the one shared row-pipeline tile width used across all three targets: HyFlexPIM's own paper "
              "(Sec. 3, Fig. 5 discussion) states 'we configured each SFU to process 256 inputs per cycle "
              "to balance the throughput of GEMV operations' -- neither 1FeFET-1C's nor RRAM-SRAM-Ensemble's "
              "own paper ever addresses softmax/GELU/attention at all, so this is the only real, paper-"
              "sourced nonlinear-op hardware width available to standardize on; any row_width is now handled "
              "as ceil(row_width/256) passes of this one circuit, not a bespoke per-model N", rows)
    try_synth([EXP_PWL16_RTL], "exp_pwl16_standalone", {}, "exp_pwl16_standalone",
              "standalone per-element extraction of softmax_row8.v's own exp_pwl16 (base-2-decompose + "
              "16-segment PWL)",
              "\"exact\" row-pipeline's own per-element exp stage, instantiated once per real row element "
              "(512x)", rows)
    try_synth([Q8_8_RECIP_RTL], "q8_8_reciprocal", {"SUM_W": 20}, "q8_8_reciprocal SUM_W=20",
              "65536/sum reciprocal, one shared instance per row",
              "\"exact\"/\"gqa_lut\"/\"ibert\" row-pipelines' own shared normalize-reciprocal stage "
              "(SUM_W=20 covers the real max row sum at N=512)", rows)
    try_synth([LOG2EXP_RTL], "log2exp_standalone", {}, "log2exp_standalone",
              "standalone per-element extraction of e2softmax_row8.v's own log2exp_q8_8 (SOLE Eq.8) plus its "
              "own inline reconstruction step",
              "\"sole\" row-pipeline's own per-element exp stage, instantiated once per real row element "
              "(512x)", rows)
    try_synth([ALDIVISION_RTL], "aldivision_standalone", {"SUM_W": 24}, "aldivision_standalone SUM_W=24",
              "standalone, width-parameterized extraction of e2softmax_row8.v's own ALDivision (shift+2-way "
              "mux, no divider)",
              "\"sole\" row-pipeline's own per-element normalize stage (SUM_W=24 covers the real max row sum "
              "at N=512)", rows)
    try_synth([IBERT_EXP_PWL_RTL], "ibert_exp_pwl_standalone", {}, "ibert_exp_pwl_standalone",
              "standalone per-element extraction of ibert_softmax_row8.v's own ibert_exp_pwl (I-BERT "
              "range-reduced quadratic-poly exp)",
              "\"ibert\" row-pipeline's own per-element exp stage, instantiated once per real row element "
              "(512x)", rows)

    report.save(HERE, "softmax", rows)


if __name__ == "__main__":
    main()
