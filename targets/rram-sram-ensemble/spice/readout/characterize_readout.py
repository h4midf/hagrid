"""The RRAM macro's read-out (Krishnan et al., TCAD 2022, Sec. III-D-1) from transistor-level
simulation, PTM 65 nm bulk, at the chip's 100 MHz. Writes default/results/readout.json.

  flash ADC    flash_tb.sp: a sweep of inputs must give monotonic codes 0 to 7, each at
               its ladder tap; energy per conversion from the supply (ngspice)
  pMOS header  sized so its resistance is header_over_min_cell_resistance of the lowest
               single-cell resistance
  read path    one read-out over one MAC (8 cycles): the header, the 8-to-1 column mux
               stepping through 8 columns, each column a resistor (rows on, at the mean
               cell conductance, each cell in series with its access transistor) and its
               wire. Measured: energy from the read supply, the part dissipated in the
               header, and the energy of the mux select drivers

Assumptions and their reasons: default/design.json "model_assumptions".

Usage (from the repo root): python3 targets/rram-sram-ensemble/spice/readout/characterize_readout.py
"""
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
from thick_model import DESIGN, MODELS, access_on_ohm, get, run  # noqa: E402

TARGET = os.path.join(HERE, "..", "..")
CELLS = os.path.join(HERE, "flash_cells.lib")
TB = os.path.join(HERE, "flash_tb.sp")
ASSUMED = DESIGN["model_assumptions"]["readout"]
MUX = DESIGN["model_assumptions"]["column_mux"]
LAYOUT = DESIGN["model_assumptions"]["array_layout"]
VDD = ASSUMED["vdd_V"]
READ_V = ASSUMED["read_V"]
MAX_HEADER_W_NM = 200000.0
TCK = 1e-6 / DESIGN["clock"]["freq_MHz"]
BITS = ("b2", "b1", "b0")
ROWS_ON = (1, 8, 16, 32, 64)
COLUMN_MUX = DESIGN["rram_macro"]["readout"]["column_mux"]
ROWS = DESIGN["rram_macro"]["array"]["rows"]


def _comparators(net: str, vref: float) -> str:
    """Each tap's comparator by its reference: PMOS inputs below mid-rail, NMOS otherwise."""
    for k in range(1, 8):
        clock, cell = ("clkb", "cmp_p") if k / 8 * vref < VDD / 2 else ("clk", "cmp_n")
        net, count = re.subn(rf"^XC{k} .*$", f"XC{k} vin r{k} {clock} t{k} t{k}b vdd {cell}", net, flags=re.M)
        if count != 1:
            raise ValueError(f"flash_tb.sp has no comparator XC{k}")
    return net


def convert(vins: list, vref: float) -> tuple:
    """Codes of the inputs, and energy per conversion (J), with the ladder spanning 0 to
    vref. Input k is held over cycle k; cycle 0 repeats the first input to warm up."""
    vins = [vins[0]] + list(vins)
    n = len(vins)
    pwl = []
    for k, v in enumerate(vins):
        t = k * TCK
        if k:
            pwl.append(f"{t - 1e-10:.6e} {vins[k - 1]}")
        pwl.append(f"{t:.6e} {v}")
    meas = [f"meas tran {b}_{k} find v({b}) at={(k + 1) * TCK - 1e-10:.6e}" for k in range(1, n) for b in BITS]
    meas.append(f"meas tran q_vdd integ i(vdd) from={TCK:.6e} to={n * TCK:.6e}")
    meas.append(f"meas tran q_vref integ i(vref) from={TCK:.6e} to={n * TCK:.6e}")
    net = _comparators(open(TB).read(), vref)
    for key, value in (("__MODELS__", MODELS), ("__CELLS__", CELLS), ("__TCK__", f"{TCK:.6e}"), ("__VDD__", str(VDD)),
                       ("__VREF__", str(vref)),
                       ("__RSEG__", f"{ASSUMED['ladder_segment_kohm'] * 1e3:.1f}"),
                       ("__CTAP__", f"{ASSUMED['ladder_tap_decoupling_fF'] * 1e-15:.3e}"), ("__PWL__", " ".join(pwl)),
                       ("__TEND__", f"{n * TCK + 1e-11:.6e}"), ("__MEAS__", "\n".join(meas))):
        net = net.replace(key, value)
    out = run(net)
    codes = [int("".join("1" if get(out, f"{b}_{k}") > VDD / 2 else "0" for b in BITS), 2) for k in range(1, n)]
    return codes, (-get(out, "q_vdd") * VDD - get(out, "q_vref") * vref) / (n - 1)


def header_point(width_nm: float, r_col: float, read_v: float) -> tuple:
    """(bitline voltage, supply current) of the divider: the enabled header above r_col."""
    out = run(f""".include {MODELS}
VDD vdd 0 {read_v}
MH bl 0 vdd vdd pmos w={width_nm}n l=65n
RC bl 0 {r_col}
.control
op
print v(bl)
print i(vdd)
.endc
.end
""")
    return get(out, "v\\(bl\\)"), -get(out, "i\\(vdd\\)")


def size_header(r_target: float, r_min_cell: float, read_v: float) -> float | None:
    """Width (nm) giving the header resistance r_target above one cell of r_min_cell, or
    None if even MAX_HEADER_W_NM cannot reach it at this read voltage."""
    lo, hi = 130.0, MAX_HEADER_W_NM
    v_bl, i = header_point(hi, r_min_cell, read_v)
    if (read_v - v_bl) / i > r_target:
        return None
    for _ in range(40):
        mid = (lo * hi) ** 0.5
        v_bl, i = header_point(mid, r_min_cell, read_v)
        if (read_v - v_bl) / i > r_target:
            lo = mid
        else:
            hi = mid
    return hi


def header_mos(width_nm: float) -> str:
    """The enabled pMOS header, gate at ground."""
    return f"MH top 0 vrd vrd pmos w={width_nm}n l=65n"


def read_path(header: str, r_col: float, read_v: float) -> dict:
    """Energies (J) of one read-out over one MAC at column resistance r_col, with the
    header netlist line `header` between the read supply (vrd) and the read node (top)."""
    tck_ns = TCK * 1e9
    mac_ns = COLUMN_MUX * tck_ns
    c_bl = ROWS * LAYOUT["cell_pitch_um"] * LAYOUT["wire_fF_per_um"]
    lines = []
    for k in range(COLUMN_MUX):
        lines += [
            f"VS{k} s{k} 0 PULSE(0 {VDD} {k * tck_ns}n 50p 50p {tck_ns - 0.05}n {mac_ns}n)",
            f"MSP{k} sb{k} s{k} vsel vsel pmos w=2080n l=65n",
            f"MSN{k} sb{k} s{k} 0 0 nmos w=1040n l=65n",
            f"MTP{k} sel{k} sb{k} vsel vsel pmos w=2080n l=65n",
            f"MTN{k} sel{k} sb{k} 0 0 nmos w=1040n l=65n",
            f"MGN{k} top sel{k} col{k} 0 nmos w={MUX['nmos_w_nm']}n l=65n",
            f"MGP{k} top sb{k} col{k} vdd pmos w={MUX['pmos_w_nm']}n l=65n",
            f"RC{k} col{k} 0 {r_col}",
            f"CB{k} col{k} 0 {c_bl}f",
        ]
    body = "\n".join(lines)
    out = run(f""".include {MODELS}
VDD vdd 0 {VDD}
VSEL vsel 0 {VDD}
VRD vrd 0 {read_v}
{header}
{body}
BH hp 0 V = (v(vrd) - v(top)) * (-i(vrd))
.tran 20p {3 * mac_ns}n
.control
run
meas tran q_vrd integ i(vrd) from={mac_ns}n to={2 * mac_ns}n
meas tran q_sel integ i(vsel) from={mac_ns}n to={2 * mac_ns}n
meas tran e_hdr integ v(hp) from={mac_ns}n to={2 * mac_ns}n
meas tran v_top avg v(top) from={mac_ns}n to={2 * mac_ns}n
.endc
.end
""")
    return {"read_supply_J": -get(out, "q_vrd") * read_v, "header_J": get(out, "e_hdr"),
            "mux_select_J": -get(out, "q_sel") * VDD, "top_V": get(out, "v_top")}


def prog_mux_leak_W(v_line: float) -> float:
    """Power leaked by one idle BL / SL programming mux (off transmission gate) with its
    array side at v_line and its programming side at ground."""
    out = run(f""".include {MODELS}
VDD vdd 0 {VDD}
VL line 0 {v_line}
MGN line 0 0 0 nmos w={MUX['nmos_w_nm']}n l=65n
MGP line vdd 0 vdd pmos w={MUX['pmos_w_nm']}n l=65n
.control
op
print i(vdd)
print i(vl)
.endc
.end
""")
    return VDD * abs(get(out, "i\\(vdd\\)")) + v_line * abs(get(out, "i\\(vl\\)"))


def mean_cell_conductance(r_access: float) -> float:
    resistances = [r * 1e3 for r in DESIGN["rram_macro"]["cell"]["resistance_kohm_approx_fig8"]]
    return sum(1 / (r + r_access) for r in resistances) / len(resistances)


def main() -> None:
    lsb = READ_V / 8
    vins = [(k + 0.5) * lsb for k in range(8)]
    codes, energy = convert(vins, READ_V)
    if codes != list(range(8)):
        raise RuntimeError(f"the flash ADC does not convert correctly at {TCK * 1e9:.0f} ns: {codes} for {vins}")
    sweep_v = [k * READ_V / 48 for k in range(1, 48)]
    sweep_codes, sweep_energy = convert(sweep_v, READ_V)
    if any(b < a for a, b in zip(sweep_codes, sweep_codes[1:])):
        raise RuntimeError(f"the flash ADC is not monotonic: {sweep_codes}")

    cell = DESIGN["rram_macro"]["cell"]
    resistances = [r * 1e3 for r in cell["resistance_kohm_approx_fig8"]]
    r_min = min(resistances)
    r_access = access_on_ohm()
    g_mean = mean_cell_conductance(r_access)
    r_target = ASSUMED["header_over_min_cell_resistance"] * r_min
    width = size_header(r_target, r_min, READ_V)
    if width is None:
        raise RuntimeError(f"no pMOS header up to {MAX_HEADER_W_NM:.0f} nm reaches {r_target:.0f} ohm at {READ_V} V")

    readouts = DESIGN["rram_macro"]["readout"]["readout_circuits"]
    mac_s = COLUMN_MUX * TCK
    columns = {}
    for n in ROWS_ON:
        p = read_path(header_mos(width), 1 / (n * g_mean), READ_V)
        columns[str(n)] = {
            "top_V": p["top_V"],
            "all_readouts_supply_mW": readouts * p["read_supply_J"] / mac_s * 1e3,
            "all_readouts_header_only_mW": readouts * p["header_J"] / mac_s * 1e3,
            "all_readouts_mux_select_mW": readouts * p["mux_select_J"] / mac_s * 1e3,
            # BL + SL muxes: the columns being read sit near the read node, the others at 0 V
            "programming_mux_leakage_mW": 2 * (readouts * prog_mux_leak_W(p["top_V"])
                                               + (ROWS - readouts) * prog_mux_leak_W(0.0)) * 1e3,
        }

    adc_power_mW = energy / TCK * 1e3
    out = {
        "vdd_V": VDD, "read_V": READ_V, "clock_ns": TCK * 1e9, "readouts": readouts,
        "flash_adc": {"codes_at_code_centers": codes, "sweep_inputs_V": sweep_v, "sweep_codes": sweep_codes,
                      "energy_pJ_per_conversion": energy * 1e12, "sweep_energy_pJ_per_conversion": sweep_energy * 1e12,
                      "power_mW": adc_power_mW, "all_readouts_power_mW": readouts * adc_power_mW},
        "header": {"width_nm": width, "target_resistance_ohm": r_target, "min_cell_ohm": r_min,
                   "access_on_ohm": r_access, "mean_cell_conductance_uS": g_mean * 1e6},
        "by_rows_on": columns,
    }
    path = os.path.join(TARGET, "default", "results", "readout.json")
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(out, f, indent=2)

    print(f"flash ADC: codes {codes}; {energy * 1e12:.3f} pJ / conversion, x{readouts} = "
          f"{readouts * adc_power_mW:.3f} mW")
    print(f"header: W = {width:.0f} nm for {r_target:.0f} ohm; access {r_access:.0f} ohm; "
          f"mean cell {g_mean * 1e6:.1f} uS")
    for n, c in columns.items():
        print(f"  {n:>2} rows on: node {c['top_V']:.3f} V, supply {c['all_readouts_supply_mW']:.2f} mW "
              f"(header only {c['all_readouts_header_only_mW']:.2f}), mux selects "
              f"{c['all_readouts_mux_select_mW']:.3f} mW, programming mux leakage "
              f"{c['programming_mux_leakage_mW']:.4f} mW")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
