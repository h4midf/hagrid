"""Synthesizes leaky_relu.v at three negative_slope values (0.01, 0.125, 0.0078125) at
both widths (Q12.20, Q8.8) and writes results/leaky_relu.json."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "leaky_relu.v")
# (width, frac, [(name, alpha), ...])
CASES = [
    (32, 20, [("0.01_generic", 10486), ("0.125_pow2", 1 << 17), ("0.0078125_pow2", 1 << 13)]),
    (16, 8, [("0.01_generic", 3), ("0.125_pow2", 1 << 5), ("0.0078125_pow2", 1 << 1)]),
]
FMT = {32: "Q12.20", 16: "Q8.8"}


def main():
    rows = []
    for w, frac, alphas in CASES:
        for name, alpha in alphas:
            res = synth.synth([RTL], "leaky_relu", {"W": w, "FRAC": frac, "ALPHA": alpha})
            row = report.row("leaky_relu", {"W": w, "alpha": name, "in": FMT[w], "out": FMT[w]},
                              "constant-multiply + mux", res, "alpha is a compile-time parameter")
            rows.append(row)
            print(f"  leaky_relu W={w} alpha={name}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps",
                  flush=True)
    report.save(HERE, "leaky_relu", rows)


if __name__ == "__main__":
    main()
