"""Functional check of rtl/rram_shift_add.v: after 6 bit-serial input bits over 8 columns,
every accumulator matches the Python model (acc = code on the first bit, then
acc = 2 acc + code).

Usage (from the repo root):
  python3 targets/rram-sram-ensemble/periphery/test_rram_shift_add.py
"""
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", "rram_shift_add.v"), os.path.join(HERE, "rtl", "rram_shift_add_tb.v")]
SEEDS = (5, 91)


def ref(seed: int) -> int:
    flat = 0
    for c in range(8):
        acc = 0
        for b in range(6):
            code = (c * 37 + b * 11 + seed) % 8
            acc = code if b == 0 else 2 * acc + code
        flat |= acc << (9 * c)
    return flat


def simulate(seed: int) -> int:
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {' '.join(RTL)}; chparam -set SEED {seed} rram_shift_add_tb; "
              f"hierarchy -top rram_shift_add_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n 52 -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    vcd = open(vcd_path).read()
    os.unlink(vcd_path)
    ident = re.search(r"\$var \w+ \d+ (\S+) acc_flat ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last:
        raise RuntimeError("acc_flat is undefined at the end of the simulation")
    return int(last, 2)


def main() -> None:
    for seed in SEEDS:
        got, want = simulate(seed), ref(seed)
        if got != want:
            raise RuntimeError(f"seed {seed}: got {got:x}, want {want:x}")
    print(f"rram_shift_add: {len(SEEDS)} seeds x 8 columns match")


if __name__ == "__main__":
    main()
