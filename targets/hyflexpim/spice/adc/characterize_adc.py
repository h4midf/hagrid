"""HyFlexPIM's ADC (Table 2: one per analog array, 7-bit, 1.28 GS/s) from transistor-level
simulation: two interleaved channels of sar_tb.sp (7-bit asynchronous SAR, PTM 65 nm),
each at half the rate. Writes default/results/adc.json.

  conversion  a sweep of single-ended inputs at the channel period: codes must be
              monotonic, include odd codes (the LSB decision completes) and have a small
              integral nonlinearity
  power       energy per conversion (VDD + Vref, ngspice) x 1.28 GS/s (both channels)
  6-bit mode  the SLC conversion (design.json adc_reconfiguration: the MSB comparison
              bypassed, _MSB_BYPASS), over half the input range: codes must be monotonic
              and span at least half of 0..63; its energy per conversion
  area        per channel: every transistor's layout area (NeuroSim's single-transistor
              layout, as targets/1fefet1c/default/area.json) + CDAC capacitors at the MOM
              density; x 2 channels
Settings and their sources: default/components.json "adc".

Usage (from the repo root): python3 targets/hyflexpim/spice/adc/characterize_adc.py
"""
import json
import os
import re
import subprocess
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "..")
MODELS = os.path.join(TARGET, "spice", "models", "65nm_bulk.pm")
CELLS = os.path.join(HERE, "sar_cells.lib")
TB = os.path.join(HERE, "sar_tb.sp")
DESIGN = json.load(open(os.path.join(TARGET, "default", "design.json")))
ADC = json.load(open(os.path.join(TARGET, "default", "components.json")))["adc"]
BITS = ["cN6", "cN5", "cN4", "cN3", "cN2", "cN1", "cN0"]

# 6-bit (SLC) mode: the comparison of the largest capacitor, decision 6, is bypassed
# (design.json analog_module.readout.adc_reconfiguration): its pointer never selects it,
# its done flag is set as soon as the conversion starts (so decision 5 comes first), and
# its CDAC capacitors never switch
_MSB_BYPASS = (
    ("Xp6 convn f6 p6 vdd nor2", "Vp6 p6 0 0"),
    ("XsF6 validA p6 sF6 vdd nand2", "Xb6 conv sF6 vdd inv"),
    ("XsN6 validA p6 dpA sN6 vdd nand3", "VsN6 sN6 0 1.0"),
    ("XsP6 validA p6 dnA sP6 vdd nand3", "VsP6 sP6 0 1.0"),
)


# ---------------------------------------------------------------- simulation
def convert(vds, tck, tsamp, vcm, bits: int):
    """Codes of single-ended inputs vin = vcm + vd (N side at vcm), and energy per
    conversion (J) drawn from VDD and from Vref, in the 7-bit mode or the 6-bit mode
    (_MSB_BYPASS; the code is then decisions 5..0). Conversion 0 (a repeat) warms up."""
    if bits not in (6, 7):
        raise ValueError(f"the ADC converts in 6- or 7-bit mode, not {bits}")
    code_bits = BITS[7 - bits:]
    vds = [vds[0]] + list(vds)
    n = len(vds)
    pwl = []
    for k, vd in enumerate(vds):
        t = (k - 1) * tck + tsamp + 100e-12 if k else 0.0     # change during the previous conversion
        if k:
            pwl.append(f"{t - 1e-12:.6e} {vcm + vds[k - 1]}")
        pwl.append(f"{t:.6e} {vcm + vd}")
    meas = [f"meas tran {b}_{k} find v({b.lower()}) at={(k + 1) * tck - 5e-12:.6e}"
            for k in range(1, n) for b in BITS]
    meas += [f"meas tran q_vdd integ i(vdd) from={tck:.6e} to={n * tck:.6e}",
             f"meas tran q_vref integ i(vref) from={tck:.6e} to={n * tck:.6e}"]
    net = open(TB).read()
    if bits == 6:
        for old, new in _MSB_BYPASS:
            if old not in net:
                raise ValueError(f"sar_tb.sp has no line {old!r} to bypass")
            net = net.replace(old, new)
    net = net.replace(".param tck=1562.5p tsamp=200p", f".param tck={tck * 1e12:.3f}p tsamp={tsamp * 1e12:.3f}p")
    net = (net.replace("__MODELS__", MODELS).replace("__CELLS__", CELLS).replace("__PWL_P__", " ".join(pwl))
           .replace("__PWL_N__", f"0 {vcm}").replace("__TEND__", f"{n * tck + 10e-12:.6e}")
           .replace("__MEAS__", "\n".join(meas)))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "adc.sp")
        with open(path, "w") as f:
            f.write(net)
        out = subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout

    def get(key):
        m = re.search(rf"^{key}\s*=\s*([-+\d.eE]+)", out, re.M)
        if not m:
            raise RuntimeError(f"ngspice: no '{key}'\n{out[-2000:]}")
        return float(m.group(1))
    codes = [int("".join("1" if get(f"{b.lower()}_{k}") > 0.5 else "0" for b in code_bits), 2) for k in range(1, n)]
    return codes, -get("q_vdd") * 1.0 / (n - 1), -get("q_vref") * 1.0 / (n - 1)


# ---------------------------------------------------------------- area
def _unbrace(line):
    return re.sub(r"\{[^}]*\}", lambda m: m.group(0).replace(" ", ""), line)


def _eval(expr, env):
    e = re.sub(r"(\d+(?:\.\d+)?)n\b", r"(\1e-9)", expr.strip("{}"))
    e = re.sub(r"(\d+(?:\.\d+)?)u\b", r"(\1e-6)", e)
    return eval(e, {"max": max}, env)


def area_um2(feature_m, mom_fF_per_um2):
    """(transistor count, transistor layout area, capacitor area) of one channel."""
    lay = ADC["transistor_layout_F"]

    def transistor(w, l):
        f = feature_m
        return (2 * (l + lay["gate_poly_gap"] * f)
                * (w + lay["p_n_diffusion_gap"] * f + 2 * (lay["poly_extension"] + lay["field_poly_gap"] / 2) * f))
    subs = {}
    for m in re.finditer(r"^\.subckt\s+(\w+)\s+([^\n]*)\n(.*?)^\.ends", open(CELLS).read(), re.S | re.M):
        toks = m.group(2).split()
        subs[m.group(1)] = ({t.split("=")[0]: t.split("=")[1] for t in toks if "=" in t},
                            [_unbrace(x) for x in m.group(3).splitlines() if x.strip() and not x.startswith("*")])

    def flat(cell, env):
        params, body = subs[cell]
        env = {**{k: _eval(v, {}) for k, v in params.items()}, **env}
        a = n = 0
        for line in body:
            t = line.split()
            if t[0][0] in "Mm":
                kv = dict(x.split("=") for x in t[6:] if "=" in x)
                a += transistor(_eval(kv["w"], env), _eval(kv["l"], env))
                n += 1
            elif t[0][0] in "Xx":
                c = next(x for x in t[1:] if x in subs)
                aa, nn = flat(c, {x.split("=")[0]: _eval(x.split("=")[1], env) for x in t[t.index(c) + 1:] if "=" in x})
                a, n = a + aa, n + nn
        return a, n

    t_area = count = cap = 0.0
    for line in open(TB):
        t = _unbrace(line).split()
        if t and t[0][0] == "X":
            c = next(x for x in t[1:] if x in subs)
            aa, nn = flat(c, {x.split("=")[0]: _eval(x.split("=")[1], {}) for x in t[t.index(c) + 1:] if "=" in x})
            t_area, count = t_area + aa, count + nn
        elif t and t[0][0] == "C":
            cap += float(t[3].rstrip("f"))
    return int(count), t_area * 1e12, cap / mom_fF_per_um2, cap


# ---------------------------------------------------------------- main
def main():
    rate = DESIGN["analog_module"]["readout"]["adc_sample_rate_GSps"] * 1e9
    ch = ADC["channels"]
    tck = ch / rate
    tsamp = ADC["sampling_window_ps"] * 1e-12
    vds = list(np.linspace(-ADC["sweep_V"], ADC["sweep_V"], 33))
    codes, e_vdd, e_vref = convert(vds, tck, tsamp, ADC["vcm_V"], 7)
    c = np.array(codes)
    inside = (c > 0) & (c < 127)
    fit = np.polyfit(np.array(vds)[inside], c[inside], 1)
    inl = c[inside] - np.polyval(fit, np.array(vds)[inside])
    conversion = {"channel_period_ps": tck * 1e12, "inputs_V": vds, "codes": codes,
                  "monotonic": bool(np.all(np.diff(c) >= 0)), "odd_codes": int(np.sum(c % 2)),
                  "gain_codes_per_V": fit[0], "inl_max_lsb": float(abs(inl).max())}
    if not (conversion["monotonic"] and conversion["odd_codes"] > 0):
        raise RuntimeError(f"the ADC does not convert correctly at {tck * 1e12:.1f} ps: {codes}")
    e = e_vdd + e_vref

    # 6-bit (SLC) mode: half the input range, codes 0..63
    vds6 = list(np.linspace(-ADC["sweep_V"] / 2, ADC["sweep_V"] / 2, 33))
    codes6, e6_vdd, e6_vref = convert(vds6, tck, tsamp, ADC["vcm_V"], 6)
    c6 = np.array(codes6)
    conversion_6bit = {"inputs_V": vds6, "codes": codes6, "monotonic": bool(np.all(np.diff(c6) >= 0)),
                       "odd_codes": int(np.sum(c6 % 2)), "code_span": [int(c6.min()), int(c6.max())]}
    if not (conversion_6bit["monotonic"] and conversion_6bit["odd_codes"] > 0 and c6.max() - c6.min() >= 32):
        raise RuntimeError(f"the ADC does not convert correctly in 6-bit mode: {codes6}")
    e6 = e6_vdd + e6_vref

    n_t, t_area, c_area, cap_fF = area_um2(DESIGN["process"]["node_nm"] * 1e-9, ADC["mom_density_fF_per_um2"])
    out = {
        "conversion": conversion,
        "energy_pJ_per_conversion": {"vdd": e_vdd * 1e12, "vref": e_vref * 1e12, "total": e * 1e12},
        "conversion_6bit": conversion_6bit,
        "energy_pJ_per_conversion_6bit": {"vdd": e6_vdd * 1e12, "vref": e6_vref * 1e12, "total": e6 * 1e12},
        "channel_area_um2": {"transistors": t_area, "capacitors": c_area, "total": t_area + c_area,
                             "transistor_count": n_t, "cdac_fF": cap_fF},
        "adc": {"channels": ch, "rate_GSps": rate * 1e-9, "power_mW": e * rate * 1e3,
                "area_um2": ch * (t_area + c_area)},
    }
    path = os.path.join(TARGET, "default", "results", "adc.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    a = out["adc"]
    print(f"channel at {tck * 1e12:.1f} ps: monotonic {conversion['monotonic']}, {conversion['odd_codes']} odd codes, "
          f"INL {conversion['inl_max_lsb']:.2f} LSB; {e * 1e12:.3f} pJ / conversion")
    print(f"6-bit mode: codes {conversion_6bit['code_span']}, monotonic; {e6 * 1e12:.3f} pJ / conversion")
    print(f"ADC ({ch} channels, {rate * 1e-9:.2f} GS/s): {a['power_mW']:.3f} mW, {a['area_um2']:.1f} um^2 "
          f"(channel: {n_t} transistors {t_area:.1f} um^2 + CDAC {cap_fF:.0f} fF {c_area:.1f} um^2)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
