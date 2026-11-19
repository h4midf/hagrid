"""Functional check of rtl/macro_ctrl.v: for every cycle of one MAC (and the cycles
after it), the phase outputs and word lines match the Python model.

Model: start in cycle 0 sets t = 1; t counts to 3 + CONV_CYCLES and returns to 0.
rst = (t == 1), mult = (t == 2), acc = (t == 3), samp = (t == 4), busy = (t != 0),
wl = the loaded input bits while mult.

Usage (from the repo root):
  python3 targets/1fefet1c/macro_logic/test_macro_ctrl.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import CONV_CYCLES, ROWS  # noqa: E402
from sim import final_value, simulate  # noqa: E402

PAT = 0xA5
LAST = 3 + CONV_CYCLES


def ref(edges: int) -> dict:
    t = edges if 1 <= edges <= LAST else 0
    pattern = int(f"{PAT:08b}" * (-(-ROWS // 8)), 2) & ((1 << ROWS) - 1)
    return {"rst": int(t == 1), "mult": int(t == 2), "acc": int(t == 3), "samp": int(t == 4),
            "busy": int(t != 0), "wl": pattern if t == 2 else 0}


def main():
    bad = []
    for edges in range(1, LAST + 3):
        vcd = simulate(["macro_ctrl", "macro_ctrl_tb"], "macro_ctrl_tb",
                       {"ROWS": ROWS, "CONV_CYCLES": CONV_CYCLES, "PAT": PAT}, edges)
        exp = ref(edges)
        got = {name: final_value(vcd, name) for name in exp}
        if got != exp:
            bad.append((edges, got, exp))
    print(f"macro_ctrl: {LAST + 2 - len(bad)}/{LAST + 2} cycles match")
    for edges, got, exp in bad[:5]:
        print(f"  after {edges} edges: rtl {got} ref {exp}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
