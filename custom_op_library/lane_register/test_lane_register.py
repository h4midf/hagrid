"""Functional check of rtl/lane_register.v: the register keeps the data of the one
cycle its load is high and ignores later data.

For every characterized width and load cycle (0 and 1), yosys simulates
rtl/lane_register_tb.v for 3 cycles and q is compared with the Python model (the
repeated byte of the loaded cycle). Exit code 0 only if every case matches.

Usage (from the repo root):
  python3 custom_op_library/lane_register/test_lane_register.py
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
sys.path.insert(0, HERE)
from characterize import BITS  # noqa: E402

PA, PB = 0xA5, 0x3C
CYCLES = 3


def ref(bits: int, load_at: int) -> int:
    byte = PA if load_at == 0 else PB
    value = 0
    for _ in range(-(-bits // 8)):
        value = (value << 8) | byte
    return value & ((1 << bits) - 1)


def final_value(vcd: str, name: str) -> int:
    ident = re.search(rf"\$var \w+ \d+ (\S+) {re.escape(name)} ", vcd).group(1)
    values = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)
    if "x" in values[-1]:
        raise RuntimeError(f"{name} is undefined at the end of the simulation")
    return int(values[-1], 2)


def simulate(bits: int, load_at: int) -> int:
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {os.path.join(RTL, 'lane_register.v')} {os.path.join(RTL, 'lane_register_tb.v')}; "
              f"chparam -set BITS {bits} -set LOAD_AT {load_at} lane_register_tb; "
              f"hierarchy -top lane_register_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n {CYCLES} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    with open(vcd_path) as f:
        vcd = f.read()
    os.unlink(vcd_path)
    return final_value(vcd, "q")


def main():
    bad = []
    cases = [(bits, load_at) for bits in BITS for load_at in (0, 1)]
    for bits, load_at in cases:
        got, exp = simulate(bits, load_at), ref(bits, load_at)
        if got != exp:
            bad.append((bits, load_at, got, exp))
    print(f"lane_register: {len(cases) - len(bad)}/{len(cases)} cases match")
    for bits, load_at, got, exp in bad:
        print(f"  BITS={bits} LOAD_AT={load_at}: rtl {got:#x} ref {exp:#x}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
