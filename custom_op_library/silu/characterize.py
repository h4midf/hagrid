"""Synthesizes silu.v at both widths and writes results/silu.json."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = [os.path.join(HERE, "rtl", "exp_pwl_q12_20.v"), os.path.join(HERE, "rtl", "exp_pwl_q8_8.v"),
       os.path.join(HERE, "rtl", "silu.v")]
CASES = [
    (32, 20, "mean abs err 9e-6, max abs err 5.4e-5 vs true SiLU"),
    (16, 8, "mean abs err 3.9e-3, max abs err 3.7e-2 vs true SiLU"),
]
FMT = {32: "Q12.20", 16: "Q8.8"}


def main():
    rows = []
    for w, frac, err_note in CASES:
        res = synth.synth(RTL, "silu", {"W": w, "FRAC": frac})
        row = report.row("silu", {"W": w, "segments": 16, "in": FMT[w], "out": FMT[w]},
                          "base-2 decompose + 16-segment piecewise-linear exp + one division", res,
                          f"validated: {err_note} (see approximation_error.py)")
        rows.append(row)
        print(f"  silu W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "silu", rows)


if __name__ == "__main__":
    main()
