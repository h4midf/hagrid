"""The lane-count and data-width sweep shared by the per-circuit characterize_*.py
scripts. Each script synthesizes one circuit and replaces only that circuit's rows in
results/vec_el_mul.json (report.save_unit), so the circuits run separately.

The file is rewritten after every point, so a run stopped by the synthesis timeout
keeps every point of this circuit finished before it.
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
COMMON = os.path.join(HERE, "..", "common", "tools")
sys.path.insert(0, COMMON)
import report  # noqa: E402
import synth  # noqa: E402

RTL = os.path.join(HERE, "rtl")
LANE_COUNTS = (1, 2, 4, 8, 16, 32, 64)
FORMATS = {32: ("Q12.20", 20), 16: ("Q8.8", 8)}


def sweep(module: str, notes: str) -> None:
    rows = []
    for n in LANE_COUNTS:
        for w, (fmt, frac) in FORMATS.items():
            res = synth.synth([os.path.join(RTL, module + ".v")], module, {"N": n, "W": w, "FRAC": frac})
            rows.append(report.row(module, {"N": n, "W": w, "FRAC": frac, "in": fmt, "out": fmt},
                                   "rescale multiply", res, notes))
            report.save_unit(HERE, "vec_el_mul", module, rows)
            print(f"  {module} N={n} W={w}: area {res['area_um2']:.0f} um^2 (45 nm), "
                  f"delay {res['delay_ps']} ps", flush=True)
