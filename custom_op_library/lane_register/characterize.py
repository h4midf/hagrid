"""Synthesizes rtl/lane_register.v at the widths the shared lanes need and writes
results/lane_register.json. BITS = input bits + result bits of one lane:

  96   vec_add / vec_el_mul lane at W = 32 (two 32-bit operands, 32-bit result)
  48   vec_add / vec_el_mul lane at W = 16
  64   gqa_lut_exp8 / quark_appro_exp at W = 32 (one operand, one result)
  32   gqa_lut_exp8 / quark_appro_exp at W = 16
  40   exp_pwl16_standalone, ibert_exp_pwl_standalone (24-bit input, 16-bit result)
  37   log2exp_standalone (24-bit input, 4 + 9-bit result)
  44   aldivision_standalone at SUM_W = 24 (4 + 24-bit input, 16-bit result)
  97   vec_select lane at W = 32 (x, fill word, result, mask bit)
  49   vec_select lane at W = 16
  58   per-token rounding lane (quant_mant_round) at W = 32 (x, 12-bit m, 6-bit sh, int8 q)
  42   per-token rounding lane at W = 16

Usage (from the repo root):
  python3 custom_op_library/lane_register/characterize.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl", "lane_register.v")
BITS = (32, 37, 40, 42, 44, 48, 49, 58, 64, 96, 97)


def main():
    rows = []
    for bits in BITS:
        res = synth.synth([RTL], "lane_register", {"BITS": bits})
        rows.append(report.row("lane_register", {"BITS": bits}, "register with load enable", res,
                               "one shared lane's input operands and result"))
        print(f"  lane_register BITS={bits}: area {res['area_um2']:.1f} um^2 (45 nm)", flush=True)
    report.save(HERE, "lane_register", rows)


if __name__ == "__main__":
    main()
