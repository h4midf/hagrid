"""Noise budget of the ADC front end, saved to results/adc.json ("noise").

  - buffer: ngspice .noise of buffer_ab as a unity-gain follower driving the sampling
    load, integrated output noise (1 Hz and 39 kHz lower limits; 39 kHz = 1/record
    length of the 64-sample ENOB test)
  - sampling kT/C on the total CDAC capacitance
Comparator noise is not simulated (dynamic circuit); run_mc.py uses a typical value.

Usage: python3 targets/1fefet1c/spice/adc/run_noise.py [ibias_uA]
"""
import re
import sys

import numpy as np

from tb import LSB, VDD, run_ngspice, save_result, testbench  # noqa: F401  (testbench: include paths)
from tb import HERE

K_B, T = 1.380649e-23, 300.0
C_SAMPLE = 129 * 2e-15        # 128 binary units + dummy, unit 2 fF (sar_full_tb.sp)
C_LOAD = 280e-15              # buffer load during sampling (sar_full_tb.sp comment)
VINS = [0.02, 0.25, 0.48]


def buffer_noise(ibias, vin, f_lo):
    net = f"""* buffer output noise, follower, sampling load
.include {HERE.parent}/models/32nm_LP.pm
.include {HERE.parent}/models/32nm_HP_lvt.pm
.include {HERE}/sar_frontend.lib
VDDA vdda 0 {VDD}
VIN bl 0 DC {vin} AC 1
Xbuf bl out vdda buffer_ab IB={ibias} CC=70f N2=4 N6=6 CLS=200f
CL out 0 {C_LOAD}
.control
noise v(out) VIN dec 50 {f_lo} 20g
setplot noise2
print onoise_total
.endc
.end
"""
    out = run_ngspice(net, f"noise_ib{ibias * 1e9:.0f}n_vin{vin}_f{f_lo:g}")
    return float(re.search(r"onoise_total\s*=\s*([-+\d.eE]+)", out).group(1))


def main():
    ibias = float(sys.argv[1]) * 1e-6 if len(sys.argv) > 1 else 0.7e-6
    buf = {str(v): {"from_1Hz_mV": buffer_noise(ibias, v, 1) * 1e3,
                    "from_39kHz_mV": buffer_noise(ibias, v, 39e3) * 1e3} for v in VINS}
    ktc = np.sqrt(K_B * T / C_SAMPLE)
    worst = max(b["from_1Hz_mV"] for b in buf.values())
    total = np.sqrt((worst * 1e-3) ** 2 + ktc ** 2)
    for v, b in buf.items():
        print(f"buffer, vin={v} V: {b['from_1Hz_mV']:.3f} mV rms (from 39 kHz: {b['from_39kHz_mV']:.3f})")
    print(f"kT/C ({C_SAMPLE * 1e15:.0f} fF): {ktc * 1e3:.3f} mV; sampling total (worst buffer): "
          f"{total * 1e3:.3f} mV = {total / LSB:.3f} LSB")
    path = save_result("adc", "noise", {
        "ibias_uA": ibias * 1e6, "load_fF": C_LOAD * 1e15, "buffer_rms_by_vin": buf,
        "ktc_rms_mV": ktc * 1e3, "sample_cap_fF": C_SAMPLE * 1e15,
        "sampling_total_rms_mV": total * 1e3, "sampling_total_rms_lsb": total / LSB,
        "lsb_mV": LSB * 1e3})
    print("->", path)


if __name__ == "__main__":
    main()
