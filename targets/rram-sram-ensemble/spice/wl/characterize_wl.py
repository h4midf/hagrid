"""One word line of the RRAM macro from transistor-level simulation: a 1.2 V input, a
level shifter to 3.3 V, a driver, and the WL load (every access transistor's gate plus
the wire). Writes default/results/wl.json: the energy of one rise and fall from each
supply. Sizes and devices: design.json "model_assumptions".

Usage (from the repo root): python3 targets/rram-sram-ensemble/spice/wl/characterize_wl.py
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from thick_model import DESIGN, MODELS, THICK, get, run  # noqa: E402

TARGET = os.path.join(HERE, "..", "..")
WL = DESIGN["model_assumptions"]["wl"]
LAYOUT = DESIGN["model_assumptions"]["array_layout"]
COLS = DESIGN["rram_macro"]["array"]["cols"]
VDD = DESIGN["model_assumptions"]["readout"]["vdd_V"]
HALF_NS = 40.0


def netlist() -> str:
    length = THICK["l_nm"]
    gates = "\n".join(f"MA{c} 0 wl 0 0 nhv w={WL['access_w_nm']}n l={length}n" for c in range(COLS))
    wire_fF = COLS * LAYOUT["cell_pitch_um"] * LAYOUT["wire_fF_per_um"]
    period = 2 * HALF_NS
    return f""".include {MODELS}
.include __THICK__
VL vl 0 {VDD}
VH vh 0 {WL['high_V']}
VIN in0 0 PULSE(0 {VDD} {HALF_NS}n 100p 100p {HALF_NS}n {period}n)
* 1.2 V buffers giving the shifter both input polarities
MP1 inb in0 vl vl pmos w=520n l=65n
MN1 inb in0 0 0 nmos w=260n l=65n
MP2 in inb vl vl pmos w=520n l=65n
MN2 in inb 0 0 nmos w=260n l=65n
* level shifter, output b follows in
MLN1 a in 0 0 nhv w={WL['level_shifter_nmos_w_nm']}n l={length}n
MLN2 b inb 0 0 nhv w={WL['level_shifter_nmos_w_nm']}n l={length}n
MLP1 a b vh vh phv w={WL['level_shifter_pmos_w_nm']}n l={WL['level_shifter_pmos_l_nm']}n
MLP2 b a vh vh phv w={WL['level_shifter_pmos_w_nm']}n l={WL['level_shifter_pmos_l_nm']}n
* driver
MD1P c b vh vh phv w=1000n l={length}n
MD1N c b 0 0 nhv w=500n l={length}n
MD2P wl c vh vh phv w={WL['driver_pmos_w_nm']}n l={length}n
MD2N wl c 0 0 nhv w={WL['driver_nmos_w_nm']}n l={length}n
{gates}
CW wl 0 {wire_fF}f
.tran 20p {3 * period}n
.control
run
meas tran wl_hi find v(wl) at={period + HALF_NS + HALF_NS * 0.9}n
meas tran wl_lo find v(wl) at={period + HALF_NS * 0.9}n
meas tran q_h integ i(vh) from={period}n to={2 * period}n
meas tran q_l integ i(vl) from={period}n to={2 * period}n
.endc
.end
"""


def main() -> None:
    out = run(netlist())
    hi, lo = get(out, "wl_hi"), get(out, "wl_lo")
    if hi < 0.95 * WL["high_V"] or lo > 0.05 * WL["high_V"]:
        raise RuntimeError(f"the WL does not swing: {lo:.3f} V to {hi:.3f} V")
    e_high = -get(out, "q_h") * WL["high_V"]
    e_low = -get(out, "q_l") * VDD
    result = {"wl_low_V": lo, "wl_high_V": hi, "energy_pJ_per_toggle_pair": {
        "high_supply": e_high * 1e12, "low_supply": e_low * 1e12, "total": (e_high + e_low) * 1e12}}
    path = os.path.join(TARGET, "default", "results", "wl.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"WL swings {lo:.3f} to {hi:.3f} V; {(e_high + e_low) * 1e12:.3f} pJ per rise and fall "
          f"({e_high * 1e12:.3f} from {WL['high_V']} V, {e_low * 1e12:.4f} from {VDD} V)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
