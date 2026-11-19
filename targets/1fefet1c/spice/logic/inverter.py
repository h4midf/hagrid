"""Leakage and speed of the per-array macro logic's transistors, in SPICE.

The macro logic (macro_logic/: input / phase control, shift-and-add datapaths, WMU
readout) is synthesized with Nangate45, whose cells use FreePDK45 low-Vt devices
(NMOS_VTL / PMOS_VTL). It runs at the 20 MHz array clock, so it can use low-power
devices instead, as the cell array does (1f1c.cfg: LSTP). This script measures, on the
same cell (Nangate's INV_X1 widths: NMOS 0.415 um, PMOS 0.63 um), with the PTM 32 nm
models our SPICE circuits already use:
  leakage : DC operating point at both inputs; power = VDD * I(VDD) + Vin * I(Vin)
            (the input source carries the gate leakage); averaged over the two
            inputs, as liberty's cell_leakage_power is
  speed   : fan-out-of-4 inverter delay (average of rise and fall)
for the low-power (32nm_LP) and high-performance (32nm_HP) models. Results:
  leakage_factor = PTM LP inverter leakage / Nangate45 INV_X1 cell_leakage_power
  slowdown       = FO4(LP) / FO4(HP): an upper bound on LP's slowdown against Nangate,
                   whose low-Vt devices we take as no faster than PTM HP

Writes results/logic.json ("inverter").
Usage: python3 targets/1fefet1c/spice/logic/inverter.py
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
sys.path.insert(0, str(HERE.parents[3] / "custom_op_library" / "common" / "tools"))
from design import save_result  # noqa: E402
from liberty import CELLS  # noqa: E402

MODELS = HERE.parent / "models"
WORK = HERE / "work"
VDD = 1.0              # PTM 32 nm nominal supply (both model cards)
L = 32e-9
WN, WP = 0.415e-6, 0.63e-6   # Nangate INV_X1 (NangateOpenCellLibrary.cdl: NMOS_VTL / PMOS_VTL)
DIFF = 0.1e-6          # drain / source diffusion length, for the junction areas
NANGATE_CELL = "INV_X1"


def run(net: str, name: str) -> str:
    WORK.mkdir(exist_ok=True)
    path = WORK / f"{name}.sp"
    path.write_text(net)
    out = subprocess.run(["ngspice", "-b", str(path)], capture_output=True, text=True)
    (WORK / f"{name}.log").write_text(out.stdout + "\n--- stderr ---\n" + out.stderr)
    return out.stdout


def value(out: str, key: str) -> float:
    hit = re.search(rf"^{key}\s*=\s*([-+\d.eE]+)", out, re.M)
    if not hit:
        raise RuntimeError(f"ngspice output has no {key}")
    return float(hit.group(1))


def mos(name, d, g, s, b, typ, w):
    return (f"{name} {d} {g} {s} {b} {typ} W={w:.4e} L={L:.4e} AD={w * DIFF:.4e} AS={w * DIFF:.4e} "
            f"PD={2 * (w + DIFF):.4e} PS={2 * (w + DIFF):.4e}")


def leakage_nW(model: str, vin: float) -> float:
    net = "\n".join([
        f"* inverter leakage, {model}, input {vin} V",
        f".include {MODELS / (model + '.pm')}",
        f"VDD vdd 0 {VDD}",
        f"VIN a 0 {vin}",
        mos("MN", "z", "a", "0", "0", "nmos", WN),
        mos("MP", "z", "a", "vdd", "vdd", "pmos", WP),
        ".op", ".control", "run",
        f"let p_leak = -i(VDD) * {VDD} - i(VIN) * {vin}",
        "print p_leak", ".endc", ".end", ""])
    return value(run(net, f"leak_{model}_in{vin:g}"), "p_leak") * 1e9


def fo4_ps(model: str) -> float:
    lines = [f"* FO4 inverter delay, {model}", f".include {MODELS / (model + '.pm')}",
             ".subckt inv a z vdd m=1",
             f"MN z a 0 0 nmos W={{{WN}*m}} L={L}", f"MP z a vdd vdd pmos W={{{WP}*m}} L={L}",
             ".ends", f"VDD vdd 0 {VDD}", f"VIN in 0 PULSE(0 {VDD} 0.1n 20p 20p 2n 4n)",
             "X1 in a vdd inv m=1", "X2 a b vdd inv m=4", "X3 b c vdd inv m=16", "X4 c d vdd inv m=64",
             ".tran 1p 4n", ".control", "run",
             f"meas tran t_rise trig v(a) val={VDD / 2} fall=1 targ v(b) val={VDD / 2} rise=1",
             f"meas tran t_fall trig v(a) val={VDD / 2} rise=1 targ v(b) val={VDD / 2} fall=1",
             ".endc", ".end", ""]
    out = run("\n".join(lines), f"fo4_{model}")
    return (value(out, "t_rise") + value(out, "t_fall")) / 2 * 1e12


def main():
    res = {}
    for model in ("32nm_LP", "32nm_HP"):
        by_input = {"0": leakage_nW(model, 0.0), "1": leakage_nW(model, VDD)}
        res[model] = {"leakage_nW_by_input": by_input,
                      "leakage_nW": sum(by_input.values()) / len(by_input),
                      "fo4_ps": fo4_ps(model)}
    nangate = CELLS[NANGATE_CELL]["leak"]
    data = {
        "cell": f"inverter, NMOS {WN * 1e6} um / PMOS {WP * 1e6} um (Nangate {NANGATE_CELL}), L {L * 1e9:.0f} nm",
        "vdd_V": VDD, "models": res,
        "nangate_cell": NANGATE_CELL, "nangate_leakage_nW": nangate,
        "leakage_factor": res["32nm_LP"]["leakage_nW"] / nangate,
        "slowdown": res["32nm_LP"]["fo4_ps"] / res["32nm_HP"]["fo4_ps"],
        "note": "leakage_factor scales Nangate45 synthesized leakage to PTM 32 nm LP devices; "
                "slowdown bounds LP's delay against Nangate's low-Vt cells (taken as no faster than PTM HP)",
    }
    path = save_result("logic", "inverter", data)
    for model, r in res.items():
        print(f"{model}: leakage {r['leakage_nW']:.4f} nW (inputs {r['leakage_nW_by_input']}), FO4 {r['fo4_ps']:.1f} ps")
    print(f"Nangate {NANGATE_CELL}: {nangate:.3f} nW -> leakage factor {data['leakage_factor']:.3e} "
          f"(1/{1 / data['leakage_factor']:.0f}), slowdown {data['slowdown']:.2f} -> {path}")


if __name__ == "__main__":
    main()
