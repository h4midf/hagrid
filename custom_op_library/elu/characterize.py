"""Synthesises elu.v (with its exp_pwl_q12_20 / exp_pwl_q8_8 primitives) at both
widths and at two alpha values each, and writes results/elu.json. The exp tables
have 16 hand-written segments at each width."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = [os.path.join(HERE, "rtl", "exp_pwl_q12_20.v"), os.path.join(HERE, "rtl", "exp_pwl_q8_8.v"),
       os.path.join(HERE, "rtl", "elu.v")]
# (width, frac, [(name, alpha), ...], error note)
CASES = [
    (32, 20, [("1.0", 1 << 20), ("0.5", 1 << 19)],
     "mean abs err 4e-6, max abs err 2.3e-4 vs true ELU"),
    (16, 8, [("1.0", 1 << 8), ("0.5", 1 << 7)],
     "mean abs err 8.5e-4, max abs err 7.5e-3 vs true ELU"),
]
FMT = {32: "Q12.20", 16: "Q8.8"}


def main():
    rows = []
    for w, frac, alphas, err_note in CASES:
        for name, alpha in alphas:
            res = synth.synth(RTL, "elu", {"W": w, "FRAC": frac, "ALPHA": alpha})
            row = report.row("elu", {"W": w, "alpha": name, "segments": 16, "in": FMT[w], "out": FMT[w]},
                              "base-2 decompose + 16-segment piecewise-linear exp", res,
                              f"validated: {err_note} (see approximation_error.py)")
            rows.append(row)
            print(f"  elu W={w} alpha={name}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps",
                  flush=True)
    report.save(HERE, "elu", rows)


if __name__ == "__main__":
    main()
