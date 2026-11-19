"""Turns a synthesis result into one JSON row and writes results/<op>.json (save: the
whole file; save_unit: only one circuit's rows, keeping the other circuits' rows).

Leakage is from the liberty file. Dynamic power is an estimate: 0.5 * alpha * C * V^2 * f,
with C the summed input-pin capacitance of all cells, alpha = 0.2, f = 1 GHz, V = 1.1 V.
"""
import json
import os

from liberty import CELLS

ALPHA, FREQ, VDD = 0.2, 1e9, 1.1
SCALE_28 = 0.35  # DeepScaleTool area factor, 45 nm -> 28 nm


def row(unit, params, method, res, notes=""):
    leak_nW = sum(CELLS[c]["leak"] * n for c, n in res["cells"].items())
    cap_fF = sum(CELLS[c]["cap"] * n for c, n in res["cells"].items())
    dyn_mW = 0.5 * ALPHA * cap_fF * 1e-15 * VDD ** 2 * FREQ * 1e3
    return {
        "unit": unit, "params": params, "method": method,
        "area_um2_45nm": round(res["area_um2"], 2),
        "area_um2_28nm": round(res["area_um2"] * SCALE_28, 2),
        "delay_ps": res["delay_ps"],
        "leakage_uW": round(leak_nW * 1e-3, 4),
        "dynamic_mW_est": round(dyn_mW, 4),
        "total_cells": sum(res["cells"].values()),
        "notes": notes,
    }


def save(op_dir, op, rows):
    out = os.path.join(op_dir, "results", f"{op}.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        json.dump(rows, f, indent=2)
    print(f"wrote {out} ({len(rows)} rows)", flush=True)


def save_unit(op_dir, op, unit, rows):
    """Replaces the rows of circuit `unit` in results/<op>.json with `rows`, keeping
    the rows of every other circuit. Starts the file if it does not exist yet."""
    out = os.path.join(op_dir, "results", f"{op}.json")
    others = []
    if os.path.exists(out):
        with open(out) as f:
            others = [r for r in json.load(f) if r["unit"] != unit]
    for r in rows:
        if r["unit"] != unit:
            raise ValueError(f"save_unit({unit!r}): got a row of {r['unit']!r}")
    save(op_dir, op, others + rows)
