"""Leakage of the chip's logic devices in SPICE, as targets/1fefet1c/spice/logic/inverter.py.

The logic is synthesized with Nangate45, whose cells are high-performance low-Vt devices,
but the chip is on a 65 nm low-power process. PTM has no 65 nm low-power model, so this
takes PTM 65 nm with its threshold set to reach the off current of the LSTP device
(design.json "model_assumptions.logic_devices") and gate tunneling off, then measures an
inverter of Nangate's INV_X1 widths scaled to 65 nm, averaged over its two inputs:

  leakage_factor = 65 nm LSTP inverter leakage / Nangate45 INV_X1 cell_leakage_power

Writes default/results/logic.json.
Usage (from the repo root): python3 targets/rram-sram-ensemble/spice/logic/inverter.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(TARGET, "..", "..", "custom_op_library", "common", "tools")]
from liberty import CELLS  # noqa: E402
from thick_model import DESIGN, MODELS, get  # noqa: E402

VDD = DESIGN["model_assumptions"]["readout"]["vdd_V"]
IOFF = DESIGN["model_assumptions"]["logic_devices"]["ioff_A_per_um"]
NANGATE_CELL = "INV_X1"
SCALE = 65 / 45
WN_NM, WP_NM = 415 * SCALE, 630 * SCALE     # Nangate INV_X1 (NMOS 0.415 um, PMOS 0.63 um)


def _set(text: str, name: str, value: str) -> str:
    new, count = re.subn(rf"(\b{name}\s*=\s*)[-+\d.eE]+", rf"\g<1>{value}", text, flags=re.I)
    if count == 0:
        raise ValueError(f"{name} not found in {MODELS}")
    return new


def model_text(vth_n: float, vth_p: float) -> str:
    text = open(MODELS).read()
    split = re.search(r"\.model\s+pmos\s+pmos", text).start()
    parts = []
    for part, vth in ((text[:split], vth_n), (text[split:], -vth_p)):
        part = _set(part, "vth0", str(vth))
        part = _set(part, "igcmod", "0")
        part = _set(part, "igbmod", "0")
        parts.append(part)
    return "".join(parts)


def run(netlist: str, vth_n: float, vth_p: float) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "m.pm")
        with open(model, "w") as f:
            f.write(model_text(vth_n, vth_p))
        path = os.path.join(tmp, "n.sp")
        with open(path, "w") as f:
            f.write(netlist.replace("__MODEL__", model))
        return subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout


def off_current(kind: str, vth: float) -> float:
    """Off current per um of a 1 um device at VDD."""
    if kind == "nmos":
        device = "M1 d 0 0 0 nmos w=1u l=65n\nVD d 0 {v}"
        out = run(f".include __MODEL__\n{device.format(v=VDD)}\n.control\nop\nprint i(vd)\n.endc\n.end\n", vth, 0.4)
    else:
        device = "M1 0 s s s pmos w=1u l=65n\nVD s 0 {v}"
        out = run(f".include __MODEL__\n{device.format(v=VDD)}\n.control\nop\nprint i(vd)\n.endc\n.end\n", 0.4, vth)
    return abs(get(out, "i\\(vd\\)"))


def fit_vth(kind: str) -> float:
    lo, hi = 0.2, 1.2
    for _ in range(40):
        mid = (lo + hi) / 2
        if off_current(kind, mid) > IOFF:
            lo = mid
        else:
            hi = mid
    return hi


def inverter_leak_nW(vth_n: float, vth_p: float, vin: float) -> float:
    out = run(f""".include __MODEL__
VDD vdd 0 {VDD}
VIN a 0 {vin}
MN z a 0 0 nmos w={WN_NM}n l=65n
MP z a vdd vdd pmos w={WP_NM}n l=65n
.control
op
print i(vdd)
.endc
.end
""", vth_n, vth_p)
    return abs(get(out, "i\\(vdd\\)")) * VDD * 1e9


def main() -> None:
    vth_n, vth_p = fit_vth("nmos"), fit_vth("pmos")
    by_input = {"0": inverter_leak_nW(vth_n, vth_p, 0.0), "1": inverter_leak_nW(vth_n, vth_p, VDD)}
    leak = sum(by_input.values()) / 2
    nangate = CELLS[NANGATE_CELL]["leak"]
    out = {"ioff_A_per_um": IOFF, "vdd_V": VDD, "vth_n_V": vth_n, "vth_p_V": vth_p, "leakage_nW_by_input": by_input,
           "leakage_nW": leak, "nangate_cell": NANGATE_CELL, "nangate_leakage_nW": nangate,
           "leakage_factor": leak / nangate}
    path = os.path.join(TARGET, "default", "results", "logic.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"fitted Vth n {vth_n:.3f} V, p {vth_p:.3f} V for {IOFF:.3g} A/um; inverter {leak:.5f} nW; Nangate "
          f"{nangate:.3f} nW; factor {out['leakage_factor']:.3e} (1/{1 / out['leakage_factor']:.0f})")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
