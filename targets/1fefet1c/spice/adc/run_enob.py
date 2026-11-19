"""ENOB and energy of the full SAR ADC (buffer + bootstrapped switch + core) from a
coherent sine test, saved to results/adc.json ("enob").

fs = one conversion per T_CONV, N conversions, input frequency = M/N * fs
(M odd, coprime with N). Codes -> FFT -> SINAD -> ENOB = (SINAD - 1.76) / 6.02.
Transient SPICE has no device noise or mismatch, so this ENOB covers quantization and
the circuit's deterministic errors (settling, nonlinearity) only; run_mc.py adds both.

Usage: python3 targets/1fefet1c/spice/adc/run_enob.py [ibias_uA] [N] [M]
"""
import sys

import numpy as np

from tb import (BIT_NODES, BITS, DESIGN, FS, FULL_TB, LALENI, T_CONV, VDD, VREF, meas, run_ngspice,
                save_result, testbench)

VMID, VAMP = 0.25, 0.24


def main():
    ibias = float(sys.argv[1]) * 1e-6 if len(sys.argv) > 1 else DESIGN["adc"]["buffer_bias_uA"] * 1e-6
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 64
    m = int(sys.argv[3]) if len(sys.argv) > 3 else 7
    fin = m / n * FS
    # conversion 0 flushes the logic; conversions 1..n are the record.
    # sample instant of conversion k is ~51 ns after its start (end of sample cycle);
    # phase-shift the sine so sample k sees phase 2*pi*m*(k-1)/n
    t_sample0 = T_CONV + 51e-9
    t_end = (n + 1) * T_CONV + 50e-9
    lines = [f"meas tran code{k} find code at={(k + 1) * T_CONV + 40e-9:.5e}" for k in range(1, n + 1)]
    lines += [f"meas tran q_{s} integ i({s.upper()}) from={T_CONV:.4e} to={(n + 1) * T_CONV:.4e}"
              for s in ("vdd", "vdda", "vref")]
    net = testbench(FULL_TB).replace(".param ibuf=1u", f".param ibuf={ibias}")
    # SIN(VO VA FREQ TD THETA PHASE[deg]); choose phase so that the sample at t_sample0 is at 0 rad
    phase_deg = (-360.0 * fin * t_sample0) % 360.0
    net = net.replace("VBL  bl   0 PWL(__PWL__)", f"VBL  bl   0 SIN({VMID} {VAMP} {fin:.6e} 0 0 {phase_deg:.6f})")
    net = net.replace("__TEND__", f"{t_end:.5e}").replace("__MEAS__", "\n".join(lines))
    # keep only the bit outputs and the supply currents (for the energy measures) in
    # memory: saving every node needs ~30 GB per 64-sample run
    net = net.replace(".control\nrun", f".save {' '.join(BIT_NODES)} i(VDD) i(VDDA) i(VREF)\n.control\nrun")
    out = run_ngspice(net, f"enob_ib{ibias * 1e9:.0f}n_N{n}_M{m}")

    codes = np.array([meas(out, f"code{k}") for k in range(1, n + 1)])
    spec = np.abs(np.fft.rfft(codes - codes.mean())) ** 2
    sinad = 10 * np.log10(spec[m] / (spec[1:].sum() - spec[m]))
    enob = (sinad - 1.76) / 6.02
    e = (-meas(out, "q_vdd") * VDD - meas(out, "q_vdda") * VDD - meas(out, "q_vref") * VREF) / n
    # reference: ideal quantizer on the same samples
    k = np.arange(n)
    ideal = np.floor((VMID + VAMP * np.sin(2 * np.pi * m * k / n)) / (VREF / 2 ** BITS))
    si = np.abs(np.fft.rfft(ideal - ideal.mean())) ** 2
    sinad_i = 10 * np.log10(si[m] / (si[1:].sum() - si[m]))
    enob_i = (sinad_i - 1.76) / 6.02

    print(f"IB={ibias * 1e6:.2f} uA, N={n}, M={m}, fin={fin / 1e3:.1f} kHz")
    print("codes:", codes.astype(int).tolist())
    print(f"SINAD = {sinad:.2f} dB, ENOB = {enob:.2f} bits (Laleni: {LALENI['adc_enob_bits']})")
    print(f"energy per conversion = {e * 1e12:.3f} pJ (VCM excluded, ~0.002 pJ)")
    print(f"ideal {BITS}-bit quantizer on the same samples: SINAD = {sinad_i:.2f} dB, ENOB = {enob_i:.2f} bits")
    path = save_result("adc", "enob", {
        "ibias_uA": ibias * 1e6, "samples": n, "cycles_in_record": m, "fin_kHz": fin / 1e3,
        "input_mid_V": VMID, "input_amp_V": VAMP, "codes": codes.astype(int).tolist(),
        "sinad_dB": sinad, "enob_bits": enob, "ideal_quantizer_enob_bits": enob_i,
        "energy_per_conversion_pJ": e * 1e12,
        "laleni_enob_bits": LALENI["adc_enob_bits"],
        "laleni_energy_per_conversion_pJ": LALENI["adc_walden_fom_fJ"] * 2 ** LALENI["adc_enob_bits"] / 1e3})
    print("->", path)


if __name__ == "__main__":
    main()
