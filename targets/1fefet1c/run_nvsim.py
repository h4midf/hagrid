"""Run NVSim on <config>/1f1c.cfg and store the array results.

NVSim (NVSim/, with the 1F1C bitline-capacitance change in SubArray.cpp) gives the
array geometry: area, bitline length and capacitance (Cpar). Its read/write numbers
are a plain memory access, not the 1F1C MAC; they are stored but not used for the MAC.

Writes:
  <config>/results/nvsim_output.txt   raw NVSim output
  <config>/results/nvsim.json         parsed numbers, next to Laleni's Cpar

Usage: python3 targets/1fefet1c/run_nvsim.py      (F1C_CONFIG selects the folder)
"""
import re
import subprocess

from f1c_io import CONFIG_DIR, LALENI, RESULTS_DIR, TARGET_DIR, save_result

NVSIM = TARGET_DIR / "NVSim" / "nvsim"


_SCALE = {"p": 1e-12, "n": 1e-9, "u": 1e-6, "m": 1e-3, "": 1.0}


def grab_unit(label, unit, text, to):
    """Value of ' - <label> = <number><prefix><unit>' converted to the prefix 'to'."""
    m = re.search(rf"^ -\s+{label} = ([\d.]+)([pnum]?){unit}\b", text, re.M)
    if not m:
        raise RuntimeError(f"NVSim output: '{label}' not found")
    return float(m.group(1)) * _SCALE[m.group(2)] / _SCALE[to]


def grab(pattern, text, cast=float):
    m = re.search(pattern, text, re.M)
    if not m:
        raise RuntimeError(f"NVSim output: '{pattern}' not found")
    return cast(m.group(1)) if m.lastindex == 1 else tuple(cast(g) for g in m.groups())


def main():
    out = subprocess.run([str(NVSIM), "1f1c.cfg"], cwd=CONFIG_DIR, capture_output=True,
                         text=True, check=True).stdout
    RESULTS_DIR.mkdir(exist_ok=True)
    (RESULTS_DIR / "nvsim_output.txt").write_text(out)

    cap = grab(r"^\[CAPBL\] numRow=(\d+) lenBitline_um=([\d.]+) capWire_fF=([\d.]+) "
               r"capAccess_fF=([\d.]+) capBitline_fF=([\d.]+)", out)
    w_um, h_um, area_um2 = grab(r"^ - Total Area = ([\d.]+)um x ([\d.]+)um = ([\d.]+)um\^2", out)
    data = {
        "cell_area_F2": grab(r"^Cell Area \(F\^2\)\s*:\s*([\d.]+)", out),
        "array_area_um2": area_um2, "array_width_um": w_um, "array_height_um": h_um,
        "area_efficiency_pct": grab(r"^ - Area Efficiency = ([\d.]+)%", out),
        "rows": int(cap[0]), "bitline_length_um": cap[1],
        "bitline_wire_cap_fF": cap[2], "bitline_switch_cap_fF": cap[3], "cpar_fF": cap[4],
        "laleni_cpar_fF": LALENI["cpar_fF"],
        "memory_read_latency_ps": grab_unit("Read Latency", "s", out, "p"),
        "memory_read_energy_pJ": grab_unit("Read Dynamic Energy", "J", out, "p"),
        "memory_write_energy_pJ": grab_unit("Write Dynamic Energy", "J", out, "p"),
        "leakage_nW": grab_unit("Leakage Power", "W", out, "n"),
        "note": "Cpar is matched by fitting the cell height (see 1f1c.cell). memory_* are a plain "
                "memory access, not the 1F1C MAC.",
    }
    path = save_result("nvsim", "array", data)
    print(f"Cpar {data['cpar_fF']:.1f} fF (Laleni {LALENI['cpar_fF']}), array {area_um2:.0f} um^2 -> {path}")


if __name__ == "__main__":
    main()
