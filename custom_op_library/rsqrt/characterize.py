"""Synthesises the rsqrt circuits on their own and writes results/rsqrt.json.

  rsqrt_q12_20  W=32 (Q12.20)   ../layer_norm/rtl/rsqrt_q12_20.v
  rsqrt_q8_8    W=16 (Q8.8)     ../layer_norm/rtl/rsqrt_q8_8.v

One evaluation per input value (a layer norm needs one per row: 1/sqrt(var + eps)).
The same circuits are inside ailayernorm_stream.v and rms_norm.v; test_rsqrt.py checks
them against their Python model.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "..", "layer_norm", "rtl")

CONFIGS = [
    # (module, params recorded in the results, notes)
    ("rsqrt_q12_20", {"W": 32, "in": "Q12.20", "out": "Q12.20"},
     "1/sqrt(x), x > 0: 2 x 32-entry seed table + one Newton iteration"),
    ("rsqrt_q8_8", {"W": 16, "in": "Q8.8", "out": "Q8.8"},
     "1/sqrt(x), x > 0: 2 x 32-entry seed table + one Newton iteration"),
]


def main():
    rows = []
    for mod, params, notes in CONFIGS:
        res = synth.synth([os.path.join(RTL, mod + ".v")], mod)
        rows.append(report.row("rsqrt", params, mod, res, notes))
        print(f"  {mod}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "rsqrt", rows)


if __name__ == "__main__":
    main()
