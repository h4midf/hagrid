"""Runs sar_full_tb.sp (buffer + bootstrapped switch + SAR core) for several buffer
bias currents in parallel. Reports, per bias:
  - sampling error: sampled bottom-plate voltage minus bitline voltage, per conversion
  - output codes
  - energy per conversion, split by source (core VDD, buffer VDDA, Vref, VCM)
and saves them to results/adc.json ("bias_sweep").

Usage: python3 targets/1fefet1c/spice/adc/run_full.py [ibias_uA ...]
"""
import sys
from concurrent.futures import ThreadPoolExecutor

from tb import FULL_TB, LSB, T_CONV, VCM, VDD, VREF, meas, run_ngspice, save_result, testbench

# bitline value for each conversion (conversion 0 flushes the logic); alternates extremes
# so the buffer must slew the full range at the start of each sample cycle
VINS = [0.25, 0.02, 0.48, 0.02, 0.48, 0.25, 0.1]


def build(ibias: float) -> str:
    pts = []
    for k, v in enumerate(VINS):
        t0 = k * T_CONV
        if k == 0:
            pts.append(f"0 {v}")
        else:
            pts.append(f"{t0 - 0.5e-9:.4e} {VINS[k - 1]}")
            pts.append(f"{t0 + 0.5e-9:.4e} {v}")
    t_end = len(VINS) * T_CONV + 50e-9
    lines = []
    for k in range(1, len(VINS)):
        t0 = k * T_CONV
        lines.append(f"meas tran samp{k} find v(vin) at={t0 + 51.5e-9:.4e}")
        lines.append(f"meas tran code{k} find code at={t0 + T_CONV + 40e-9:.4e}")
    a, b = 1 * T_CONV, len(VINS) * T_CONV
    for src in ("VDD", "VDDA", "VREF", "VCM"):
        lines.append(f"meas tran q_{src.lower()} integ i({src}) from={a:.4e} to={b:.4e}")
    net = testbench(FULL_TB).replace(".param ibuf=1u", f".param ibuf={ibias}")
    net = net.replace("__PWL__", " ".join(pts)).replace("__TEND__", f"{t_end:.4e}")
    return net.replace("__MEAS__", "\n".join(lines))


def run(ibias: float) -> dict:
    out = run_ngspice(build(ibias), f"full_ib{ibias * 1e9:.0f}n")
    n = len(VINS) - 1
    res = {"ibias_uA": ibias * 1e6,
           "samp_err_lsb": [(meas(out, f"samp{k}") - VINS[k]) / LSB for k in range(1, len(VINS))],
           "codes": [meas(out, f"code{k}") for k in range(1, len(VINS))]}
    for src, v in (("vdd", VDD), ("vdda", VDD), ("vref", VREF), ("vcm", VCM)):
        res[f"e_{src}_pJ"] = -meas(out, f"q_{src}") * v / n * 1e12
    res["e_total_pJ"] = res["e_vdd_pJ"] + res["e_vdda_pJ"] + res["e_vref_pJ"] + res["e_vcm_pJ"]
    return res


if __name__ == "__main__":
    ibs = [float(x) * 1e-6 for x in sys.argv[1:]] or [0.25e-6, 0.5e-6, 1e-6, 2e-6, 4e-6]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, ibs))
    ideal = [int(v / LSB) for v in VINS[1:]]
    print("bitline per conversion:", VINS[1:], " ideal codes:", ideal)
    for r in results:
        errs = " ".join(f"{e:+5.2f}" for e in r["samp_err_lsb"])
        print(f"IB={r['ibias_uA']:5.2f}uA | sampling error [LSB]: {errs} | "
              f"codes {[int(c) if c == c else None for c in r['codes']]}")
        print(f"      energy/conv: core VDD {r['e_vdd_pJ']:.3f} pJ, buffer {r['e_vdda_pJ']:.3f} pJ, "
              f"Vref {r['e_vref_pJ']:.3f} pJ, VCM {r['e_vcm_pJ']:.3f} pJ  -> total {r['e_total_pJ']:.3f} pJ")
    path = save_result("adc", "bias_sweep", {
        "bitline_inputs_V": VINS[1:], "ideal_codes": ideal, "by_bias": results})
    print("->", path)
