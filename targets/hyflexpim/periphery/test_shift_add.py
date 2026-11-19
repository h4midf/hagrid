"""Functional check of rtl/hfp_shift_add.v in SLC and MLC mode: after 8 bit-serial
input bits (each one array read of 128 codes, one per cycle), every output's
accumulator in the modeled output register matches the Python model.

Model: the code of column j at bit b is (j * 37 + b * 11 + SEED) mod 2^CODE (CODE 6 for
SLC, 7 for MLC); output o's partial at bit b is sum_k code(o * CELLS + k, b) * W^k
(SLC: 8 cells, W = 2; MLC: 4 cells, W = 4); acc = -partial at bit 0 (the sign bit),
then acc = 2 acc + partial.

Usage (from the repo root):
  python3 targets/hyflexpim/periphery/test_shift_add.py
"""
import os
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", "hfp_shift_add.v"), os.path.join(HERE, "rtl", "hfp_shift_add_tb.v"),
       os.path.join(HERE, "..", "sfu", "rtl", "sfu_prefix_add.v")]
ACC = 22
CYCLES = 8 * 128 + 6          # the stream plus the pipeline drain
SEEDS = (5, 91)


def ref(mode, seed):
    code_bits, cells, w = (7, 4, 4) if mode else (6, 8, 2)
    outs = 128 // cells
    accs = []
    for o in range(outs):
        acc = 0
        for b in range(8):
            partial = sum((((o * cells + k) * 37 + b * 11 + seed) % (1 << code_bits)) * w ** k
                          for k in range(cells))
            acc = -partial if b == 0 else 2 * acc + partial
        accs.append(acc)
    return accs


def simulate(mode, seed):
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {' '.join(RTL)}; chparam -set MODE {mode} -set SEED {seed} hfp_shift_add_tb; "
              f"hierarchy -top hfp_shift_add_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n {CYCLES} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    vcd = open(vcd_path).read()
    os.unlink(vcd_path)
    ident = re.search(r"\$var \w+ \d+ (\S+) acc_flat ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last:
        raise RuntimeError("acc_flat is undefined at the end of the simulation")
    return int(last, 2)


def main():
    bad, total = [], 0
    for mode in (0, 1):
        for seed in SEEDS:
            flat = simulate(mode, seed)
            for o, exp in enumerate(ref(mode, seed)):
                raw = (flat >> (o * ACC)) & ((1 << ACC) - 1)
                got = raw - (1 << ACC) if raw >> (ACC - 1) else raw
                total += 1
                if got != exp:
                    bad.append((mode, seed, o, got, exp))
    print(f"hfp_shift_add: {total - len(bad)}/{total} outputs match (SLC and MLC)")
    for mode, seed, o, got, exp in bad[:5]:
        print(f"  {'MLC' if mode else 'SLC'} seed {seed} output {o}: rtl {got} ref {exp}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
