"""Synthesises every dequantise variant at both characterized output
widths (XW=32/Q12.20, XW=16/Q8.8) and writes results/dequantize.json.

Mirror of quantize/characterize.py, direction reversed (int8 in, fixed-
point out).
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl")
MANT_WIDTHS = [8, 12, 16, 24, 31]
XWS = {32: "Q12.20", 16: "Q8.8"}

VARIANTS = [
    # (module, uses mantissa, uses zero point, rounding, scale)
    ("dequant_mant_round", True, False, "round to nearest", "mantissa"),
    ("dequant_mant_trunc", True, False, "truncate", "mantissa"),
    ("dequant_pow2_round", False, False, "round to nearest", "power of two"),
    ("dequant_pow2_trunc", False, False, "truncate", "power of two"),
    ("dequant_mant_round_zp", True, True, "round to nearest", "mantissa"),
    ("dequant_pow2_round_zp", False, True, "round to nearest", "power of two"),
]


def main():
    rows = []
    for xw, fmt in XWS.items():
        for mod, has_mant, has_zp, rounding, scale in VARIANTS:
            widths = MANT_WIDTHS if has_mant else [None]
            for mw in widths:
                params = {"XW": xw}
                if has_mant:
                    params["MW"] = mw
                res = synth.synth([os.path.join(RTL, mod + ".v")], mod, params)
                notes = f"{scale} scale, {rounding}" + (", zero point" if has_zp else "")
                rows.append(report.row(mod, {"XW": xw, "MW": mw, "sh_bits": 6, "in": "int8", "out": fmt},
                                       "direct", res, notes))
                print(f"  {mod} XW={xw} MW={mw}: area {res['area_um2']:.0f} um^2, delay {res['delay_ps']} ps",
                      flush=True)
    report.save(HERE, "dequantize", rows)


if __name__ == "__main__":
    main()
