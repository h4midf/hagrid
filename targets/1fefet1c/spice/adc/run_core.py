"""Runs sar_core_tb.sp (SAR core with an ideal input) at several DC inputs in parallel,
reports codes and energy per conversion, and saves them to results/adc.json ("core_dc").

Usage: python3 targets/1fefet1c/spice/adc/run_core.py [vin ...]
"""
import sys
from concurrent.futures import ThreadPoolExecutor

from tb import LSB, T_CONV, VCM, VDD, VREF, meas, run_ngspice, save_result, testbench

N_CONV = 3  # three conversions of the same input


def run(vin: float) -> dict:
    meas_lines = [f"meas tran conv{k} find code1 at={(k + 1) * T_CONV + 40e-9:.5e}" for k in range(N_CONV)]
    # energy over one full conversion (the second one)
    a, b = T_CONV, 2 * T_CONV
    meas_lines += [f"meas tran q_{s} integ i({s.upper()}) from={a:.5e} to={b:.5e}" for s in ("vdd", "vref", "vcm", "vin")]
    net = testbench("sar_core_tb.sp").replace(".param vin_dc=0.25", f".param vin_dc={vin}")
    net = net.replace("__TEND__", f"{(N_CONV + 0.125) * T_CONV:.5e}").replace("__MEAS__", "\n".join(meas_lines))
    out = run_ngspice(net, f"core_vin{vin:.4f}")
    return {
        "vin_V": vin,
        "ideal_code": vin / LSB,
        "codes": [int(meas(out, f"conv{k}")) for k in range(N_CONV)],
        # energy drawn from each source over one conversion
        "e_vdd_fJ": -meas(out, "q_vdd") * VDD * 1e15,
        "e_vref_fJ": -meas(out, "q_vref") * VREF * 1e15,
        "e_vcm_fJ": -meas(out, "q_vcm") * VCM * 1e15,
        "e_vin_fJ": -meas(out, "q_vin") * vin * 1e15,
    }


if __name__ == "__main__":
    vins = [float(v) for v in sys.argv[1:]] or [0.02, 0.0625, 0.125, 0.1875, 0.25, 0.3125, 0.375, 0.4375, 0.48]
    with ThreadPoolExecutor(max_workers=8) as ex:
        results = list(ex.map(run, vins))
    print(f"{'vin':>7} {'ideal':>6} {'codes':>16} {'E_vdd':>8} {'E_vref':>8} {'E_vcm':>8} {'E_vin':>8} {'E_total':>8}  (fJ)")
    for r in results:
        r["e_total_fJ"] = r["e_vdd_fJ"] + r["e_vref_fJ"] + r["e_vcm_fJ"]
        print(f"{r['vin_V']:7.4f} {r['ideal_code']:6.1f} {str(r['codes']):>16} "
              f"{r['e_vdd_fJ']:8.1f} {r['e_vref_fJ']:8.1f} {r['e_vcm_fJ']:8.1f} {r['e_vin_fJ']:8.1f} {r['e_total_fJ']:8.1f}")
    path = save_result("adc", "core_dc", {
        "conversion_ns": T_CONV * 1e9, "lsb_mV": LSB * 1e3, "by_vin": results,
        "core_energy_range_pJ": [min(r["e_total_fJ"] for r in results) / 1e3,
                                 max(r["e_total_fJ"] for r in results) / 1e3]})
    print("->", path)
