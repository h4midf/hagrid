"""Dynamic power of the SRAM macro's logic from gate-level simulation of its mapped NanGate45
netlist over one macro pass, replacing the flow's flat 20% activity. Writes
results/sram_logic_power.json.

Same method as targets/hyflexpim/sfu/power_sfu.py: every cell input pin's toggles are
weighted by its liberty capacitance, energy = toggles x 0.5 C V^2 (V = 1.1 V). Zero-delay
simulation: no glitches, no cell internal power, no wire capacitance (all only add). All
flops are clocked every cycle (enables are muxes; no clock gating is assumed).

Stimulus: rtl/rse_sram_logic_tb.v, about half the input and weight bits at 1. Measured over
the macro pass only, not the input load. The mapped netlist is checked against the Python
model first.

Usage (from the repo root, after characterize_sram_macro.py):
  python3 targets/rram-sram-ensemble/periphery/power_sram_logic.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools")
sys.path[:0] = [TOOLS, os.path.join(HERE, "..", "..", "hyflexpim", "sfu"), HERE]
import synth  # noqa: E402
from liberty import LIB  # noqa: E402
from power_sfu import CAPS, VDD, instances  # noqa: E402
from scaling import DESIGN, factors, sram_logic_rtl  # noqa: E402
from test_rse_sram_logic import CYCLES, ref  # noqa: E402

TOP = "rse_sram_logic"
PASS_START, PASS_END = 70, 236   # cycles of the macro pass in the testbench
SEED = 5


def mapped_netlist(path: str) -> None:
    """The mapped netlist of scaling.sram_logic_rtl(), by the same yosys / ABC commands as
    synth.synth(sized=False)."""
    script = (f"read_verilog -sv {sram_logic_rtl()}; hierarchy -top {TOP}; synth -flatten -top {TOP}; "
              f"dfflibmap -liberty {LIB}; techmap -map {synth._LATCH_MAP}; "
              f"abc -liberty {LIB} -script {synth._ABC_SCRIPT}; opt_clean; "
              f"write_verilog -noattr {path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def simulate(mapped: str, workdir: str) -> str:
    tb = open(os.path.join(HERE, "rtl", TOP + "_tb.v")).read().replace(f"module {TOP}_tb", "module tb")
    with open(os.path.join(workdir, "tb.v"), "w") as f:
        f.write(tb)
    script = (f"read_liberty -ignore_miss_func {LIB}; read_verilog {mapped}; read_verilog -sv tb.v; "
              f"chparam -set SEED {SEED} tb; hierarchy -top tb; proc; memory; flatten; "
              f"sim -clock clk -zinit -n {CYCLES} -vcd sim.vcd")
    subprocess.run(["yosys", "-q", "-p", script], cwd=workdir, check=True)
    return os.path.join(workdir, "sim.vcd")


def check_output(vcd_path: str) -> None:
    vcd = open(vcd_path).read()
    ident = re.search(r"\$var \w+ \d+ (\S+) out_flat ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last or int(last, 2) != ref(SEED):
        raise RuntimeError("the mapped netlist does not match the Python model")


def toggle_energy(mapped: str, vcd_path: str) -> tuple:
    """(logic pJ, clock pJ, cycles) over the macro pass."""
    inst = instances(mapped)
    names, last, changes = defaultdict(list), {}, defaultdict(list)
    clk_id = None
    t = 0
    with open(vcd_path) as f:
        for line in f:
            if line.startswith("$var"):
                p = line.split()
                if p[2] == "1" and p[4] == "clk" and clk_id is None:
                    clk_id = p[3]
                if p[2] == "1" and p[4].startswith("dut."):
                    names[p[3]].append(p[4][4:])
            elif line.startswith("#"):
                t = int(line[1:])
            elif line.startswith("b") and " " in line:
                v, i = line[1:].split()
                if i in last and last[i] != v:
                    changes[i].append(t)
                last[i] = v
    if clk_id is None:
        raise RuntimeError("no clk in the VCD")
    rising = changes[clk_id][::2]
    t0, t1 = rising[PASS_START], rising[PASS_END]
    logic = clock = 0.0
    matched = 0
    for i, pins in names.items():
        n = sum(1 for x in changes[i] if t0 <= x < t1)
        for name in pins:
            if "." not in name:
                continue
            cell, pin = name.rsplit(".", 1)
            cap = CAPS.get((inst.get(cell), pin))
            if cap is None:
                continue
            matched += 1
            e = 0.5 * cap * 1e-15 * VDD ** 2 * n * 1e12
            if pin == "CK":
                clock += e
            else:
                logic += e
    if matched == 0:
        raise RuntimeError("no cell pins matched in the VCD")
    return logic, clock, PASS_END - PASS_START


def main() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        mapped = os.path.join(tmp, "mapped.v")
        mapped_netlist(mapped)
        vcd = simulate(mapped, tmp)
        check_output(vcd)
        logic, clock, cycles = toggle_energy(mapped, vcd)

    f = factors()
    clock_GHz = DESIGN["clock"]["freq_MHz"] * 1e-3
    row = json.load(open(os.path.join(HERE, "results", "sram_logic.json")))[0]["chip_65nm"]
    logic_mW = logic / cycles * f["energy"] * clock_GHz
    clock_mW = clock / cycles * f["energy"] * clock_GHz
    dynamic = logic_mW + clock_mW
    out = {"cycles": cycles, "logic_pJ_per_cycle_45nm": logic / cycles, "clock_pJ_per_cycle_45nm": clock / cycles,
           "logic_mW": logic_mW, "clock_mW": clock_mW, "dynamic_mW": dynamic,
           "flat_activity_dynamic_mW": row["dynamic_mW"], "leakage_mW": row["leakage_mW"],
           "power_mW": dynamic + row["leakage_mW"]}
    path = os.path.join(HERE, "results", "sram_logic_power.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"mapped netlist correct; {cycles} cycles")
    print(f"dynamic {dynamic:.4f} mW (logic {logic_mW:.4f} + clock {clock_mW:.4f}), was {row['dynamic_mW']:.4f} at flat "
          f"20%; leakage {row['leakage_mW']:.4f}; total {out['power_mW']:.4f} mW")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
