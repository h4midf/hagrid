"""Functional check of the config's shift-and-add (F1C_CONFIG; rtl/shift_add.v, or
rtl/shift_add_flip.v when design.json has ISAAC's weight-flip encoding): after 8
bit-serial input bits, every output's accumulator matches the Python model.

Model: the code of column j at bit b is v = (j * 37 + b * 11 + SEED) mod 2^CODE; with
the encoding (codes an array can produce), the unit column's code is
U = (b * 13 + SEED) mod (ROWS + 1), column j is flipped when (j * 7 + SEED) mod 3 == 0,
its code is then v mod (3 U + 1) and its value 3 U - code.
Output o's partial at bit b is sum_k value(o * CELLS + k, b) << (2k); acc = -partial at
bit 0 (the sign bit), then acc = 2 * acc + partial.

Usage (from the repo root):
  F1C_CONFIG=128 python3 targets/1fefet1c/macro_logic/test_shift_add.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from params import (ACC, BITS_PER_CELL, CELLS, CODE, CONFIG, CV, ENCODED, INPUT_BITS, OUTS, PART,  # noqa: E402
                    ROWS, UNITS)
from sim import final_value, simulate  # noqa: E402

SEEDS = (5, 91)


def value(j: int, b: int, seed: int) -> int:
    code = (j * 37 + b * 11 + seed) % (1 << CODE)
    if ENCODED and (j * 7 + seed) % 3 == 0:
        top = ((1 << BITS_PER_CELL) - 1) * ((b * 13 + seed) % (ROWS + 1))
        return top - code % (top + 1)
    return code


def ref(seed: int) -> list[int]:
    accs = []
    for o in range(OUTS):
        acc = 0
        for b in range(INPUT_BITS):
            partial = sum(value(o * CELLS + k, b, seed) << (BITS_PER_CELL * k) for k in range(CELLS))
            acc = -partial if b == 0 else 2 * acc + partial
        accs.append(acc)
    return accs


def main():
    bad = []
    if ENCODED:
        modules, top = ["shift_add_flip", "shift_add_flip_tb"], "shift_add_flip_tb"
        params = {"OUTS": OUTS, "CELLS": CELLS, "CODE": CODE, "CV": CV, "PART": PART, "ACC": ACC, "UNITS": UNITS,
                  "ROWS": ROWS}
        cycles = 1 + INPUT_BITS * OUTS // UNITS   # cycle 0 loads the flip bits
    else:
        modules, top = ["shift_add", "shift_add_tb"], "shift_add_tb"
        params = {"OUTS": OUTS, "CELLS": CELLS, "CODE": CODE, "PART": PART, "ACC": ACC, "UNITS": UNITS}
        cycles = INPUT_BITS * OUTS // UNITS
    for seed in SEEDS:
        vcd = simulate(modules, top, {**params, "SEED": seed}, cycles)
        flat = final_value(vcd, "acc_flat")
        for o, exp in enumerate(ref(seed)):
            raw = (flat >> (o * ACC)) & ((1 << ACC) - 1)
            got = raw - (1 << ACC) if raw >> (ACC - 1) else raw
            if got != exp:
                bad.append((seed, o, got, exp))
    total = len(SEEDS) * OUTS
    print(f"{top[:-3]} ({CONFIG}): {total - len(bad)}/{total} outputs match")
    for seed, o, got, exp in bad[:5]:
        print(f"  seed {seed} output {o}: rtl {got} ref {exp}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
