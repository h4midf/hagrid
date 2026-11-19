"""Sizes the ADC input buffer for a resolution above the default's 7 bits.

The 8-bit SAR (sar_full_tb_8b.sp) keeps the 7-bit unit capacitor, so the sampling DAC
the buffer drives is 2x the default's, and the LSB is 2x smaller: the default buffer (CC=70f, N2=4, N6=6 at 0.7 uA) rings on the
larger load and misses the criterion. This sweeps the class-AB buffer's bias current
(IB), compensation cap (CC), output PMOS (N2) and output NMOS (N6) sizes, with the
level-shift cap (CLS=200f) and bootstrapped switch (CB=25f) of the default, on the
config's full testbench (tb.FULL_TB) with run_full.py's stimulus (full-range bitline
steps at the start of every sample cycle).

A point passes with the default's criterion: every sample within 0.5 LSB (the config's
LSB) of the bitline. The design is the passing point with the lowest energy per
conversion (core VDD + buffer VDDA + Vref); if no point passes, the point with the
smallest worst sample error, provided it is under 1 LSB (MAX_SAMPLE_ERROR_LSB_RELAXED),
else the script fails. Saves results/adc.json ("frontend_resize")
with every point and the chosen one; the chosen sizes then go into the config's
testbench Xbuf line and its bias into design.json adc.buffer_bias_uA.

Usage: F1C_CONFIG=128 python3 targets/1fefet1c/spice/adc/run_frontend_resize.py
"""
import itertools
import math
import os
from concurrent.futures import ThreadPoolExecutor

from run_full import VINS, build
from tb import BITS, FULL_TB, LSB, VDD, VREF, WORK, meas, run_ngspice, save_result

TB_BUF = "Xbuf bl vbuf vdda buffer_ab IB={ibuf} CC=70f N2=4 N6=6 CLS=200f"   # the testbench's buffer line
MAX_SAMPLE_ERROR_LSB = 0.5
# our decision when no point meets MAX_SAMPLE_ERROR_LSB: the point with the smallest
# worst sample error, if it is under this many LSB
MAX_SAMPLE_ERROR_LSB_RELAXED = 1.0

IB_UA = (0.7, 1.0, 1.5, 2.0, 3.0)
# pruned from the first 8-bit points at 0.7 uA: CC = 70f rings (up to 11 LSB), and an
# output NMOS much larger than the PMOS (N6 = 24, or N6 = 12 with N2 = 4) leaves a
# negative error of 2-5 LSB; kept: N6 / N2 = 1.5 (the default's) and 12 / 8
CC_FF = (100, 150, 200)
N2_N6 = ((4, 6), (8, 12))
# fine grid: at 8 bits no point of the grid above passed; CC = 70f rings and CC >= 150f
# is slew-limited, more bias raises the error near ground, and the closest point was
# 0.7 uA, CC = 100f (0.72 LSB): CC between 70f and 100f at low bias
FINE_IB_UA = (0.5, 0.6, 0.7, 0.8)
FINE_CC_FF = (75, 80, 85, 90)
POINTS = ([{"IB_uA": ib, "CC": cc, "N2": n2, "N6": n6}
           for ib, cc, (n2, n6) in itertools.product(IB_UA, CC_FF, N2_N6)]
          + [{"IB_uA": ib, "CC": cc, "N2": n2, "N6": n6}
             for ib, cc, (n2, n6) in itertools.product(FINE_IB_UA, FINE_CC_FF, N2_N6)])


def _netlist(point: dict) -> str:
    net = build(point["IB_uA"] * 1e-6)
    if TB_BUF not in net:
        raise RuntimeError(f"{FULL_TB} no longer has the line {TB_BUF!r}")
    buf = f"Xbuf bl vbuf vdda buffer_ab IB={{ibuf}} CC={point['CC']}f N2={point['N2']} N6={point['N6']} CLS=200f"
    return net.replace(TB_BUF, buf)


def run(point: dict) -> dict:
    name = f"resize{BITS}b_" + "_".join(f"{k}{v}" for k, v in point.items())
    # a point whose run finished earlier (its log has every measure) is read, not rerun
    log = WORK / f"{name}.log"
    earlier = log.read_text() if log.exists() else ""
    if not math.isnan(meas(earlier, f"code{len(VINS) - 1}")):
        out = earlier
    else:
        out = run_ngspice(_netlist(point), name)
    n = len(VINS) - 1
    errs = [(meas(out, f"samp{k}") - VINS[k]) / LSB for k in range(1, len(VINS))]
    codes = [meas(out, f"code{k}") for k in range(1, len(VINS))]
    energy = -sum(meas(out, f"q_{s}") * v for s, v in (("vdd", VDD), ("vdda", VDD), ("vref", VREF))) / n * 1e12
    ok = all(e == e for e in errs) and max(abs(e) for e in errs) < MAX_SAMPLE_ERROR_LSB
    return {**point, "samp_err_lsb": errs, "codes": codes, "energy_per_conversion_pJ": energy, "passes": ok}


if __name__ == "__main__":
    if BITS <= 7:
        raise SystemExit(f"{BITS}-bit ADC: the default front end (run_frontend_sweep.py) applies")
    with ThreadPoolExecutor(max_workers=os.cpu_count()) as ex:
        results = list(ex.map(run, POINTS))
    for r in sorted(results, key=lambda r: (not r["passes"], r["energy_per_conversion_pJ"])):
        worst = max(abs(e) for e in r["samp_err_lsb"]) if all(e == e for e in r["samp_err_lsb"]) else float("nan")
        print(f"{'PASS' if r['passes'] else 'fail'} IB={r['IB_uA']:3.1f}uA CC={r['CC']:3d}f N2={r['N2']} N6={r['N6']:2d} | "
              f"worst {worst:6.2f} LSB | {r['energy_per_conversion_pJ']:6.3f} pJ | "
              f"codes {[int(c) if c == c else None for c in r['codes']]}")
    passing = [r for r in results if r["passes"]]
    if passing:
        best = min(passing, key=lambda r: r["energy_per_conversion_pJ"])
        criterion = f"every sample within {MAX_SAMPLE_ERROR_LSB} LSB, lowest energy"
    else:
        def worst(r):
            return max(abs(e) for e in r["samp_err_lsb"]) if all(e == e for e in r["samp_err_lsb"]) else math.inf
        best = min(results, key=worst)
        if worst(best) >= MAX_SAMPLE_ERROR_LSB_RELAXED:
            raise SystemExit(f"no point within {MAX_SAMPLE_ERROR_LSB_RELAXED} LSB at {BITS} bits "
                             f"(closest {worst(best):.2f} LSB)")
        criterion = (f"no point within {MAX_SAMPLE_ERROR_LSB} LSB: the smallest worst sample error "
                     f"({worst(best):.2f} LSB, under {MAX_SAMPLE_ERROR_LSB_RELAXED} LSB)")
    path = save_result("adc", "frontend_resize", {
        "bits": BITS, "testbench": FULL_TB, "bitline_inputs_V": VINS[1:],
        "ideal_codes": [int(v / LSB) for v in VINS[1:]], "max_sample_error_lsb": MAX_SAMPLE_ERROR_LSB,
        "max_sample_error_lsb_relaxed": MAX_SAMPLE_ERROR_LSB_RELAXED, "criterion": criterion,
        "fixed": {"CLS_fF": 200, "CB_fF": 25}, "chosen": best, "points": results})
    print(f"chosen ({criterion}): IB={best['IB_uA']} uA CC={best['CC']}f N2={best['N2']} N6={best['N6']} "
          f"({best['energy_per_conversion_pJ']:.3f} pJ) -> {path}")
