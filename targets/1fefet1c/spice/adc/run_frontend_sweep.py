"""Sweeps the ADC front end (input buffer + bootstrapped switch) for the smallest
capacitors that still sample correctly, at the buffer bias used for all ADC results.

Each point runs sar_full_tb.sp (run_full.py's stimulus: full-range bitline steps at
the start of every sample cycle) with the testbench's buffer and bootstrapped-switch
lines replaced:
  class-AB buffer (buffer_ab)  CLS (level-shift cap) x CC (compensation) x CB (bootstrap)
  Miller buffer (buffer)       CC x N2 (second-stage size) x CB, nulling resistor RZ
A point passes when every sample is within 0.5 LSB of the bitline (the criterion the
current sizes were chosen with) and every output code is within 1 of the current
design's code at the same input. The front end's capacitance (the area that matters,
area.json column_frontend) is reported per point.

Saves results/adc.json ("frontend_sweep").

Usage: python3 targets/1fefet1c/spice/adc/run_frontend_sweep.py
"""
import itertools
import os
from concurrent.futures import ThreadPoolExecutor

from run_full import VINS, build
from tb import LSB, meas, run_ngspice, save_result

IBIAS = 0.7e-6   # buffer bias used for all ADC results (spice/mac/rmu_mac.py IBUF)
CURRENT_BUF = "Xbuf bl vbuf vdda buffer_ab IB={ibuf} CC=70f N2=4 N6=6 CLS=200f"
CURRENT_BS = "Xbs vbuf vin samp sampb vdd bootsw CB=25f"
MAX_SAMPLE_ERROR_LSB = 0.5
MAX_CODE_DIFF = 1

CB_FF = (100, 50, 25)
POINTS = (
    [{"buffer": "buffer_ab", "CLS": cls, "CC": cc, "CB": cb}
     for cls, cc, cb in itertools.product((200, 100, 50, 25), (70, 50, 35, 25), CB_FF)]
    + [{"buffer": "buffer", "CC": cc, "N2": n2, "CB": cb}
       for cc, n2, cb in itertools.product((100, 70, 50, 35), (4, 8), CB_FF)]
)


def _netlist(point: dict) -> str:
    net = build(IBIAS)
    for line in (CURRENT_BUF, CURRENT_BS):
        if line not in net:
            raise RuntimeError(f"sar_full_tb.sp no longer has the line {line!r}")
    if point["buffer"] == "buffer_ab":
        buf = f"Xbuf bl vbuf vdda buffer_ab IB={{ibuf}} CC={point['CC']}f N2=4 N6=6 CLS={point['CLS']}f"
    else:
        buf = f"Xbuf bl vbuf vdda buffer IB={{ibuf}} CC={point['CC']}f RZ=20k N2={point['N2']}"
    return net.replace(CURRENT_BUF, buf).replace(CURRENT_BS, f"Xbs vbuf vin samp sampb vdd bootsw CB={point['CB']}f")


def _name(point: dict) -> str:
    return "fe_" + "_".join(f"{k}{v}" for k, v in point.items())


def run(point: dict) -> dict:
    out = run_ngspice(_netlist(point), _name(point))
    n = len(VINS) - 1
    errs = [(meas(out, f"samp{k}") - VINS[k]) / LSB for k in range(1, len(VINS))]
    codes = [meas(out, f"code{k}") for k in range(1, len(VINS))]
    energy = -sum(meas(out, f"q_{s}") * v for s, v in (("vdd", 1.0), ("vdda", 1.0))) / n * 1e12
    return {**point, "capacitance_fF": point["CC"] + point.get("CLS", 0) + point["CB"],
            "samp_err_lsb": errs, "codes": codes, "e_vdd_vdda_pJ": energy}


if __name__ == "__main__":
    current = run({"buffer": "buffer_ab", "CLS": 200, "CC": 70, "CB": 25})
    if any(c != c for c in current["codes"]) or max(abs(e) for e in current["samp_err_lsb"]) >= MAX_SAMPLE_ERROR_LSB:
        raise RuntimeError(f"the current front end does not pass its own criterion: {current}")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(run, POINTS))
    for r in results:
        r["passes"] = (all(c == c for c in r["codes"])
                       and max(abs(e) for e in r["samp_err_lsb"]) < MAX_SAMPLE_ERROR_LSB
                       and all(abs(c - c0) <= MAX_CODE_DIFF for c, c0 in zip(r["codes"], current["codes"])))
    for r in sorted(results, key=lambda r: (not r["passes"], r["capacitance_fF"])):
        knobs = " ".join(f"{k}={v}" for k, v in r.items() if k in ("buffer", "CLS", "CC", "N2", "CB"))
        worst = max(abs(e) for e in r["samp_err_lsb"]) if all(e == e for e in r["samp_err_lsb"]) else float("nan")
        print(f"{'PASS' if r['passes'] else 'fail'} {r['capacitance_fF']:4d} fF | {knobs:45s} | "
              f"worst sample error {worst:5.2f} LSB | codes {[int(c) if c == c else None for c in r['codes']]}")
    path = save_result("adc", "frontend_sweep", {
        "ibias_uA": IBIAS * 1e6, "bitline_inputs_V": VINS[1:], "current": current,
        "max_sample_error_lsb": MAX_SAMPLE_ERROR_LSB, "max_code_diff": MAX_CODE_DIFF, "points": results})
    print("->", path)
