"""Synthesizes vec_el_mul_vs_round (vector*scalar broadcast, round to nearest; the
multiply by a constant that bert_base_int8_emulator.py models) over sweep.py's lane
counts and data widths.

Usage (from the repo root):
  python3 custom_op_library/vec_el_mul/characterize_vs_round.py
"""
from sweep import sweep

if __name__ == "__main__":
    sweep("vec_el_mul_vs_round", "vector*scalar (broadcast), round to nearest")
