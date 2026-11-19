"""Functional check of rtl/sar_logic.v against a Python model of the SAR search.

For every input code 0..127, yosys simulates one conversion of rtl/sar_logic_tb.v
(ideal comparator, 8 cycles) and the final bit register is compared with the Python
model: from the MSB down, keep a trial bit when the DAC level with it is at most the
input. Exit code 0 only if all 128 codes match.

Usage (from the repo root):
  python3 targets/1fefet1c/adc_logic/test_sar_logic.py
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
BITS = 7
CYCLES = 8  # one sample cycle, then one trial cycle per bit


def ref(code: int) -> int:
    b = 0
    for k in reversed(range(BITS)):
        trial = b | (1 << k)
        if trial <= code:
            b = trial
    return b


def final_value(vcd: str, name: str) -> int:
    ident = re.search(rf"\$var \w+ \d+ (\S+) {re.escape(name)} ", vcd).group(1)
    values = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)
    if "x" in values[-1]:
        raise RuntimeError(f"{name} is undefined at the end of the simulation")
    return int(values[-1], 2)


def simulate(code: int) -> int:
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {os.path.join(RTL, 'sar_logic.v')} {os.path.join(RTL, 'sar_logic_tb.v')}; "
              f"chparam -set CODE {code} sar_logic_tb; hierarchy -top sar_logic_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n {CYCLES} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    with open(vcd_path) as f:
        vcd = f.read()
    os.unlink(vcd_path)
    return final_value(vcd, "b")


def main():
    bad = []
    for code in range(1 << BITS):
        got, exp = simulate(code), ref(code)
        if got != exp:
            bad.append((code, got, exp))
    print(f"sar_logic: {(1 << BITS) - len(bad)}/{1 << BITS} codes match")
    for code, got, exp in bad[:5]:
        print(f"  code {code}: rtl {got} ref {exp}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
