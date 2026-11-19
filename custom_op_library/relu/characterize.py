"""Synthesizes relu.v and shifted_relu.v (HAMMER's exp replacement) at both widths
(W=32/Q12.20, W=16/Q8.8) and writes results/relu.json."""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "relu.v")
SHIFTED_RTL = os.path.join(HERE, "rtl", "shifted_relu.v")
WIDTHS = {32: "Q12.20", 16: "Q8.8"}
FRACS = {32: 20, 16: 8}


def main():
    rows = []
    for w, fmt in WIDTHS.items():
        res = synth.synth([RTL], "relu", {"W": w})
        row = report.row("relu", {"W": w, "in": fmt, "out": fmt}, "exact (comparator/mux)", res,
                          "y = max(x, 0); exact, not an approximation")
        rows.append(row)
        print(f"  relu W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    for w, fmt in WIDTHS.items():
        res = synth.synth([SHIFTED_RTL], "shifted_relu", {"W": w, "FRAC": FRACS[w]})
        row = report.row("shifted_relu", {"W": w, "in": fmt, "out": fmt}, "4-term shifted-ReLU adder network", res,
                          "HAMMER (IEEE CAL 2023) exp-replacement; own reconstruction, no closed-form in source")
        rows.append(row)
        print(f"  shifted_relu W={w}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "relu", rows)


if __name__ == "__main__":
    main()
