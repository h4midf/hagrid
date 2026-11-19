"""Design point and results I/O for the SPICE scripts (see targets/1fefet1c/f1c_io.py).

The config folder is taken from the F1C_CONFIG environment variable, default "default":
    F1C_CONFIG=128 python3 targets/1fefet1c/spice/column/column_sim.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from f1c_io import CONFIG, DESIGN, LALENI, T_CLK, load_result, save_result  # noqa: E402,F401
