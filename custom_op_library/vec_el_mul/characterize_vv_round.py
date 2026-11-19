"""Synthesizes vec_el_mul_vv_round (vector*vector, round to nearest; the compiler's
vec_el_mul) over sweep.py's lane counts and data widths.

Usage (from the repo root):
  python3 custom_op_library/vec_el_mul/characterize_vv_round.py
"""
from sweep import sweep

if __name__ == "__main__":
    sweep("vec_el_mul_vv_round", "vector*vector, round to nearest")
