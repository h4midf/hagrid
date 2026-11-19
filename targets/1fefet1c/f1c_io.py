"""Shared I/O for the 1F1C target scripts: the design point and the results files.

Config folder (default / 64 / 128) comes from the F1C_CONFIG environment
variable, default "default":
    design:  targets/1fefet1c/<config>/design.json
    results: targets/1fefet1c/<config>/results/<name>.json

save_result(name, key, data) updates one key of results/<name>.json (other keys
are kept), so each experiment can be rerun on its own.
"""
import json
import os
import sys
from datetime import datetime
from pathlib import Path

TARGET_DIR = Path(__file__).resolve().parent
CONFIG = os.environ.get("F1C_CONFIG", "default")
CONFIG_DIR = TARGET_DIR / CONFIG
RESULTS_DIR = CONFIG_DIR / "results"
DESIGN = json.loads((CONFIG_DIR / "design.json").read_text())
LALENI = DESIGN["reported_results"]

T_CLK = 1e-6 / DESIGN["clock"]["freq_MHz"]   # one clock cycle [s]


def save_result(name: str, key: str, data: dict) -> Path:
    RESULTS_DIR.mkdir(exist_ok=True)
    path = RESULTS_DIR / f"{name}.json"
    doc = json.loads(path.read_text()) if path.exists() else {}
    doc[key] = {"generated": datetime.now().isoformat(timespec="seconds"),
                "script": Path(sys.argv[0]).name, **data}
    path.write_text(json.dumps(doc, indent=2) + "\n")
    return path


def load_result(name: str) -> dict:
    path = RESULTS_DIR / f"{name}.json"
    return json.loads(path.read_text()) if path.exists() else {}
