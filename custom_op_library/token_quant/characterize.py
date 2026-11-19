"""Synthesises the per-token quantisation units and writes results/token_quant.json.

  absmax_reduce_tree  W=32 (Q12.20), N=256: the reduction width of softmax's
                      max_reduce_tree; a 768- / 3072-wide row is 3 / 12 passes
  token_scale         W=32, MW=12: the mantissa width quant_mant_round is used at

The per-element rounding is ../quantize's quant_mant_round (results/quantize.json).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl")

CONFIGS = [
    # (module, params, notes)
    ("absmax_reduce_tree", {"N": 256, "W": 32},
     "max |x| over N signed lanes, one pass; unsigned W-bit output"),
    ("token_scale", {"W": 32, "MW": 12, "SH_MAX": 31},
     "m, sh of 127 / M for quant_mant_round; one per row (combinational divide)"),
]


def main():
    rows = []
    for mod, params, notes in CONFIGS:
        res = synth.synth([os.path.join(RTL, mod + ".v")], mod, params)
        rows.append(report.row(mod, params, "direct", res, notes))
        print(f"  {mod} {params}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "token_quant", rows)


if __name__ == "__main__":
    main()
