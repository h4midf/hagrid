"""MAC from the FeFETs (RMU MAC), in SPICE: the 128-cell column (with its bitline wire)
driving the SAR ADC (input buffer + bootstrapped switch + core) directly.

Timing (one MAC = 1 load cycle + one ADC conversion):
  the column's reset and multiply phases run in the cycles before conversion 1;
  its accumulate phase (charge sharing onto the bitline) starts with conversion 1,
  whose first cycle samples the bitline. Conversion 0 only flushes the ADC logic.

For each test case three simulations:
  combined   : column -> ADC                                 (the real read path)
  ideal_input: ADC driven by an ideal source at the column-only bitline voltage
  column_only: the column alone (no ADC load), as in column_sim.py
combined vs ideal_input isolates what connecting the real column changes
(bitline loading by the ADC input, sampling during charge sharing).

Writes results/mac.json ("rmu_mac").
Usage: python3 targets/1fefet1c/spice/mac/rmu_mac.py
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "column"))
sys.path.insert(0, str(HERE.parent / "adc"))
import column_sim as col  # noqa: E402
import tb as adc  # noqa: E402

LSB = adc.LSB
T_CONV = adc.T_CONV
T0 = T_CONV - col.T_P3          # column start so its accumulate phase begins with conversion 1
T_SAMPLE_END = T_CONV + 51e-9   # top plate opens at tclk + 1 ns
T_CODE = 2 * T_CONV + 40e-9     # conversion 1 result valid
T_END = 2 * T_CONV + 50e-9
# MAC latency: from the start of the multiply phase (FeFET -> CMoM load) to the end of
# conversion 1 (code final). The reset phase is not counted: it runs before the load.
LATENCY = 2 * T_CONV - (T0 + col.T_P2)
BITS = adc.BIT_NODES


def adc_netlist(source_lines: list[str], name: str) -> str:
    net = adc.testbench(adc.FULL_TB).replace(".param ibuf=1u", f".param ibuf={IBUF}")
    net = net.replace("VBL  bl   0 PWL(__PWL__)", "\n".join([f"* bitline source: {name}"] + source_lines))
    meas = [f"meas tran vbl_at_sample find v(bl) at={T_SAMPLE_END - 1e-9:.6e}",
            f"meas tran vin_sampled find v(vin) at={T_SAMPLE_END + 0.5e-9:.6e}",
            f"meas tran code find code at={T_CODE:.6e}"]
    net = net.replace("__TEND__", f"{T_END:.6e}").replace("__MEAS__", "\n".join(meas))
    return net.replace(".control\nrun", f".save v(bl) v(vin) {' '.join(BITS)}\n.control\nrun")


IBUF = adc.DESIGN["adc"]["buffer_bias_uA"] * 1e-6   # buffer bias used for all ADC results (design.json)


def run_case(case):
    name, vths, inputs = case["name"], case["vths"], case["inputs"]
    ref = col.simulate(vths, inputs, f"mac_colonly_{name}", with_wire=True)
    v_col = ref["vbl"]
    # combined: column bottom node bl0 tied to the ADC input node bl
    col_lines = col.column_circuit(vths, inputs, with_wire=True, t0=T0) + ["VLINK bl0 bl 0"]
    o_c = adc.run_ngspice(adc_netlist(col_lines, "column"), f"mac_combined_{name}")
    o_i = adc.run_ngspice(adc_netlist([f"VBL bl 0 {v_col:.6f}"], "ideal"), f"mac_ideal_{name}")
    r = {"name": name, "mac_value": case["mac"],
         "vbl_column_only_V": v_col,
         "vbl_combined_at_sample_V": adc.meas(o_c, "vbl_at_sample"),
         "sampled_combined_V": adc.meas(o_c, "vin_sampled"),
         "sampled_ideal_V": adc.meas(o_i, "vin_sampled"),
         "code_combined": adc.meas(o_c, "code"), "code_ideal_input": adc.meas(o_i, "code"),
         "ideal_code_from_vbl": v_col / LSB}
    r["bitline_loading_mV"] = (r["vbl_combined_at_sample_V"] - v_col) * 1e3
    r["code_difference"] = r["code_combined"] - r["code_ideal_input"]
    return r


def cases():
    n = col.N_ROWS
    lo, off = min(col.VTH_STATES), max(col.VTH_STATES)
    weight_of = {v: w for w, v in enumerate(sorted(col.VTH_STATES_LINEAR, reverse=True))}
    out = []
    for k in (0, n // 4, n // 2, n):
        out.append({"name": f"lowvth_{k}", "vths": [lo] * k + [off] * (n - k), "inputs": [1] * n,
                    "mac": f"{k} cells at Vth {lo} V"})
    rng = random.Random(1)
    vths = [rng.choice(col.VTH_STATES_LINEAR) for _ in range(n)]
    inputs = [rng.randint(0, 1) for _ in range(n)]
    out.append({"name": "random_linear", "vths": vths, "inputs": inputs,
                "mac": sum(weight_of[v] * x for v, x in zip(vths, inputs))})
    return out


def main():
    cs = cases()
    # calibrate every Vth state once, sequentially: delvto_for writes one shared probe file
    for v in sorted({v for c in cs for v in c["vths"]}):
        col.delvto_for(v)
    with ThreadPoolExecutor(max_workers=len(cs)) as ex:
        results = list(ex.map(run_case, cs))
    print(f"{'case':>15} {'VBL col':>8} {'VBL comb':>9} {'load mV':>8} {'code ideal':>10} {'code comb':>9}")
    for r in results:
        print(f"{r['name']:>15} {r['vbl_column_only_V']:8.4f} {r['vbl_combined_at_sample_V']:9.4f} "
              f"{r['bitline_loading_mV']:8.2f} {r['code_ideal_input']:10.0f} {r['code_combined']:9.0f}")
    path = adc.save_result("mac", "rmu_mac", {
        "latency_ns": LATENCY * 1e9,
        "buffer_bias_uA": IBUF * 1e6, "lsb_mV": LSB * 1e3, "column_start_ns": T0 * 1e9,
        "sample_end_ns": T_SAMPLE_END * 1e9, "code_at_ns": T_CODE * 1e9, "cases": results,
        "max_abs_code_difference": max(abs(r["code_difference"]) for r in results),
        "max_abs_bitline_loading_mV": max(abs(r["bitline_loading_mV"]) for r in results)})
    print("->", path)


if __name__ == "__main__":
    main()
