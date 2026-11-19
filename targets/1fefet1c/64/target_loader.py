"""Target description of the 1F1C config with 64 x 64 arrays (design.json): see
targets/1fefet1c/sized_loader.py."""
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from sized_loader import load_config  # noqa: E402


def load():
    return load_config(HERE)
