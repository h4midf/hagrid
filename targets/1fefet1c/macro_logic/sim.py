"""Shared by the macro_logic tests: run a test harness in yosys for a number of clock
cycles and read a signal's final value from the waveform."""
import os
import re
import subprocess
import tempfile

RTL = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rtl")


def simulate(modules: list[str], top: str, params: dict, cycles: int) -> str:
    """The VCD text of `cycles` clock cycles of `top` (rtl/<module>.v for each module)."""
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    files = " ".join(os.path.join(RTL, m + ".v") for m in modules)
    chparam = " ".join(f"-set {k} {v}" for k, v in params.items())
    script = (f"read_verilog -sv {files}; chparam {chparam} {top}; hierarchy -top {top}; "
              f"proc; flatten; opt_clean; sim -clock clk -zinit -n {cycles} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    with open(vcd_path) as f:
        vcd = f.read()
    os.unlink(vcd_path)
    return vcd


def final_value(vcd: str, name: str) -> int:
    ident = re.search(rf"\$var \w+ \d+ (\S+) {re.escape(name)} ", vcd).group(1)
    values = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)
    if "x" in values[-1]:
        raise RuntimeError(f"{name} is undefined at the end of the simulation")
    return int(values[-1], 2)
