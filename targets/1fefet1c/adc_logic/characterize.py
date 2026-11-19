"""Synthesizes rtl/sar_logic.v (the digital part of one column's 7-bit SAR ADC) with
the shared yosys / Nangate45 flow (custom_op_library/common/tools/synth.py) and
writes results/sar_logic.json: area at 45 nm and rescaled to 28 nm, the same way as
the accelerator units.

Usage (from the repo root):
  python3 targets/1fefet1c/adc_logic/characterize.py
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

MODULE = "sar_logic"

if __name__ == "__main__":
    res = synth.synth([os.path.join(HERE, "rtl", MODULE + ".v")], MODULE)
    row = report.row(MODULE, {"BITS": 7}, "SAR logic: shift register, bit registers, DAC switch control", res,
                     "digital part of one column's 7-bit SAR ADC (Laleni et al. 2024, Fig. 9)")
    report.save(HERE, MODULE, [row])
    print(f"  {MODULE}: area {res['area_um2']:.1f} um^2 (45 nm), {row['area_um2_28nm']} um^2 (28 nm), "
          f"cells {res['cells']}", flush=True)
