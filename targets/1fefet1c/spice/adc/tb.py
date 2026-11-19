"""Shared setup for the SAR ADC scripts: design values and testbench loading.

Design values (supply, Vref, bits, clock, cycles per conversion) come from
<config>/design.json (see targets/1fefet1c/f1c_io.py). The testbenches carry a
__DESIGN__ line that is replaced by a .param line with these values.

VCM (comparator common mode) and the unit capacitor are our implementation choices,
not Laleni values, so they stay in the netlists.
"""
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from design import CONFIG, DESIGN, LALENI, T_CLK, load_result, save_result  # noqa: E402,F401

WORK = HERE / "work"

VDD = DESIGN["process"]["vdd_V"]
VREF = DESIGN["adc"]["vrefp_V"]
BITS = DESIGN["adc"]["bits"]
CYCLES = DESIGN["adc"]["cycles_per_conversion"]
T_CONV = CYCLES * T_CLK
FS = 1 / T_CONV
LSB = VREF / 2 ** BITS
VCM = 0.3   # our choice: low VCM so the top-plate switch turns on fully (LP Vth ~0.56 V)

# the full ADC testbench (buffer + bootstrapped switch + core) of each resolution
FULL_TBS = {7: "sar_full_tb.sp", 8: "sar_full_tb_8b.sp"}
if BITS not in FULL_TBS:
    raise SystemExit(f"{CONFIG}: design.json asks for a {BITS}-bit ADC; full testbenches exist for {sorted(FULL_TBS)}")
FULL_TB = FULL_TBS[BITS]
BIT_NODES = [f"v(b{i})" for i in range(BITS)]   # the decided bits, MSB b{BITS-1}


def testbench(name: str) -> str:
    """Testbench text with absolute include paths and the design .param line."""
    net = (HERE / name).read_text()
    net = net.replace(".include ../models/", f".include {HERE.parent}/models/")
    for lib in ("sar_core.lib", "sar_frontend.lib"):
        net = net.replace(f".include {lib}", f".include {HERE}/{lib}")
    return net.replace("__DESIGN__", f".param vdd_v={VDD} vref_v={VREF} tclk={T_CLK:.6e} tconv={T_CONV:.6e}")


def run_ngspice(net: str, name: str) -> str:
    """Writes work/<name>.sp, runs it, keeps stdout+stderr in work/<name>.log, returns stdout."""
    WORK.mkdir(exist_ok=True)
    path = WORK / f"{name}.sp"
    path.write_text(net)
    o = subprocess.run(["ngspice", "-b", str(path)], capture_output=True, text=True)
    (WORK / f"{name}.log").write_text(o.stdout + "\n--- stderr ---\n" + o.stderr)
    return o.stdout


def meas(out: str, key: str) -> float:
    hit = re.search(rf"^{key}\s*=\s*([-+\d.eE]+)", out, re.M)
    return float(hit.group(1)) if hit else float("nan")
