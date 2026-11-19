"""Functional check of the SRAM macro logic (scaling.sram_logic_rtl()): after one macro pass, each of the 16 PEs' output
matches the Python model, the sum over its 42 signed 3-bit weights of weight x input bit.

Usage (from the repo root):
  python3 targets/rram-sram-ensemble/periphery/test_rse_sram_logic.py
"""
import os
import re
import subprocess
import tempfile

from scaling import sram_logic_rtl

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [sram_logic_rtl(), os.path.join(HERE, "rtl", "rse_sram_logic_tb.v")]
W, ACC, NW, PES, CYCLES = 3, 9, 42, 16, 260
SEEDS = (5, 91)


def ref(seed: int) -> int:
    flat = 0
    for p in range(PES):
        acc = 0
        for k in range(NW):
            value = 0
            for b in range(W):
                value = (value << 1) | (((k * W + b) * 13 + p * 7 + seed) >> 2 & 1)
            weight = value - (1 << W) if value >> (W - 1) else value
            acc += weight * ((k * 5 + seed) >> 1 & 1)
        flat |= (acc & ((1 << ACC) - 1)) << (p * ACC)
    return flat


def simulate(seed: int) -> int:
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {' '.join(RTL)}; chparam -set SEED {seed} rse_sram_logic_tb; "
              f"hierarchy -top rse_sram_logic_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n {CYCLES} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    vcd = open(vcd_path).read()
    os.unlink(vcd_path)
    ident = re.search(r"\$var \w+ \d+ (\S+) out_flat ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last:
        raise RuntimeError("out_flat is undefined at the end of the simulation")
    return int(last, 2)


def main() -> None:
    for seed in SEEDS:
        got, want = simulate(seed), ref(seed)
        if got != want:
            raise RuntimeError(f"seed {seed}: got {got:x}, want {want:x}")
    print(f"rse_sram_logic: {len(SEEDS)} seeds x {PES} PEs match")


if __name__ == "__main__":
    main()
