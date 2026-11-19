"""Synthesizes rms_norm.v at N=4, at both widths, and writes results/rms_norm.json.
Only N=4: each lane holds 3 WxW multipliers, and ABC's runtime grows super-linearly
with replicated multipliers.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = [os.path.join(HERE, "rtl", "reduce_tree.v"), os.path.join(HERE, "rtl", "rsqrt_q12_20.v"),
       os.path.join(HERE, "rtl", "rsqrt_q8_8.v"), os.path.join(HERE, "rtl", "rms_norm.v")]
N = 4
CASES = [
    (32, 20, "Q12.20", "mean abs err 3.0e-6, max abs err 3.0e-5 vs true RMSNorm"),
    (16, 8, "Q8.8", "mean abs err 2.1e-2, max abs err 1.3e-1 (uniform [-10,10]); clustered "
                    "(realistic) case: mean abs err 6.3e-3, max abs err 5.7e-2"),
]


def main():
    rows = []
    for w, frac, fmt, err_note in CASES:
        res = synth.synth(RTL, "rms_norm", {"N": N, "W": w, "FRAC": frac})
        row = report.row("rms_norm", {"N": N, "W": w, "in": fmt, "out": fmt},
                          "mean-square reduction + sqrt-via-rsqrt + reciprocal-via-rsqrt + gamma scale", res,
                          f"validated: {err_note} (see approximation_error.py)")
        rows.append(row)
        print(f"  rms_norm W={w} N={N}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps", flush=True)
    report.save(HERE, "rms_norm", rows)


if __name__ == "__main__":
    main()
