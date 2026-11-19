"""SPICE simulation of one 1F1C column (Laleni et al. 2024, Fig. 4 / Fig. 9).

Cell i:
  FeFET   : source on SL (= VDD), gate on WL_i (= input), drain on node X_i.
            Modeled as a PTM 32nm LP NMOS, W/L = 450/450 nm (Laleni Sec. III-A),
            with its threshold set to the programmed state via delvto.
  CMoM    : 1.3 fF from X_i to ground (Laleni Sec. IV).
  Switch  : CMOS transmission gate between X_i and the bitline (minimum size).
Bitline:    distributed RC, one segment per row, per-row R/C from NVSim
            (default/1f1c.cfg + 1f1c.cell: local aggressive wire, 1.896 um/row).
Reset:      NMOS at the bottom of the bitline (discharges BL and, with S on, all CMoMs).

Design values (VDD, node, CMoM, FeFET W/L, rows, Vth states, clock and phase cycles) are
read from targets/1fefet1c/<config>/design.json (config = F1C_CONFIG, default "default").

Phases (one clock cycle each at 20 MHz for the default = 50 ns; Laleni Sec. III-B / Fig. 4):
  P1   0- 50 ns  precharge: RST=1, S=1 -> CMoMs and BL discharged to 0; WL=0
  P2  50-100 ns  input:     S=0, WL_i = input_i * VDD -> X_i charges to ~VDD - Vth_i
  P3 100-150 ns  accumulate: RST=0, S=1 -> charge sharing on BL; VBL read at 149 ns

Usage: python3 targets/1fefet1c/spice/column/column_sim.py
"""
import itertools
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))
from design import DESIGN, LALENI, T_CLK, load_result, save_result  # noqa: E402  (<config>/design.json, results/)

MODEL = HERE.parent / "models" / "32nm_LP.pm"
WORK = HERE / "work"

VDD = DESIGN["process"]["vdd_V"]
F = DESIGN["process"]["node_nm"] * 1e-9
C_MOM = DESIGN["cell"]["cmom_fF"] * 1e-15
FEFET_W, FEFET_L = DESIGN["cell"]["fefet_w_nm"] * 1e-9, DESIGN["cell"]["fefet_l_nm"] * 1e-9
N_ROWS = DESIGN["array"]["rows"]
VTH_STATES = DESIGN["cell"]["vth_states_V"]
VTH_STATES_LINEAR = DESIGN["cell"]["vth_states_linear_V"]
SW_W, SW_L = 48e-9, 32e-9          # minimum-size switch devices (PTM 32 nm)
RST_W, RST_L = 96e-9, 32e-9
CONTACT = 2 * F                    # source/drain diffusion length for junction caps

# Per-row bitline wire, from NVSim (results/nvsim.json, written by run_nvsim.py):
#   C = bitline wire capacitance / rows; R = 27.87 ohm/um (NVSim local aggressive
#   wire at 28 nm; NVSim does not print it) x bitline length / rows
_NV = load_result("nvsim")["array"]
ROW_C = _NV["bitline_wire_cap_fF"] * 1e-15 / _NV["rows"]
ROW_R = 27.87 * _NV["bitline_length_um"] / _NV["rows"]

# phase boundaries: reset, multiply, accumulate, each a whole number of clock cycles
_PH = DESIGN["clock"]["column_phase_cycles"]
T_P1 = 0.0
T_P2 = T_P1 + _PH["reset"] * T_CLK
T_P3 = T_P2 + _PH["multiply"] * T_CLK
T_END = T_P3 + _PH["accumulate"] * T_CLK
EDGE = 0.2e-9


def run_ngspice(netlist: str, name: str) -> str:
    WORK.mkdir(exist_ok=True)
    path = WORK / f"{name}.sp"
    path.write_text(netlist)
    out = subprocess.run(["ngspice", "-b", str(path)], capture_output=True, text=True)
    return out.stdout + out.stderr


def meas(output: str, key: str) -> float:
    m = re.search(rf"^{key}\s*=\s*([-+\d.eE]+)", output, re.M)
    if not m:
        raise RuntimeError(f"measurement {key} not found:\n{output[-3000:]}")
    return float(m.group(1))


# ---------------------------------------------------------------- Vth calibration
_DELVTO_CACHE: dict[float, float] = {}


def delvto_for(vth: float) -> float:
    """delvto that gives constant-current Vth (100 nA * W/L, Vd = 50 mV) = vth."""
    if vth in _DELVTO_CACHE:
        return _DELVTO_CACHE[vth]

    def measured(d: float) -> float:
        net = f"""* vth probe
.include {MODEL}
M1 d g 0 0 nmos W={FEFET_W} L={FEFET_L} delvto={d}
Vd d 0 0.05
Vg g 0 0
.dc Vg -0.5 2.0 0.0005
.control
run
meas dc vth when i(Vd)=-100n
.endc
.end
"""
        return meas(run_ngspice(net, "vth_probe"), "vth")

    lo, hi = -1.0, 1.0
    for _ in range(25):
        mid = (lo + hi) / 2
        if measured(mid) < vth:
            lo = mid
        else:
            hi = mid
    _DELVTO_CACHE[vth] = (lo + hi) / 2
    return _DELVTO_CACHE[vth]


# ---------------------------------------------------------------- netlist
def pwl(points):
    return "PWL(" + " ".join(f"{t:.4e} {v:.4f}" for t, v in points) + ")"


def column_circuit(vths, inputs, with_wire: bool, t0: float) -> list[str]:
    """Element lines of one column (no .include/.tran), phases starting at t0:
    reset [t0, t0+T_P2], multiply [t0+T_P2, t0+T_P3], accumulate [t0+T_P3, t0+T_END].
    Before t0 the column sits in its reset state. Bitline bottom node: bl0."""
    n = len(vths)
    ad_fe = FEFET_W * CONTACT
    pd_fe = 2 * (FEFET_W + CONTACT)
    ad_sw = SW_W * CONTACT
    pd_sw = 2 * (SW_W + CONTACT)

    def shifted(pts):
        out = [(t + t0, v) for t, v in pts]
        return [(0, pts[0][1])] + out if t0 > 0 else out

    lines = []
    # supplies
    lines.append(f"VSL sl 0 {VDD}")
    # S high in P1 and P3, low in P2; Sb is its complement
    s_pts = [(0, VDD), (T_P2, VDD), (T_P2 + EDGE, 0), (T_P3, 0), (T_P3 + EDGE, VDD), (T_END, VDD)]
    lines.append(f"VS s 0 {pwl(shifted(s_pts))}")
    lines.append(f"VSB sb 0 {pwl(shifted([(t, VDD - v) for t, v in s_pts]))}")
    # RST high through P1 and P2, off just before P3
    r_pts = [(0, VDD), (T_P3 - 1e-9, VDD), (T_P3 - 1e-9 + EDGE, 0), (T_END, 0)]
    lines.append(f"VRST rst 0 {pwl(shifted(r_pts))}")

    for i, (vth, x) in enumerate(zip(vths, inputs)):
        d = delvto_for(vth)
        w_pts = [(0, 0), (T_P2 + 1e-9, 0), (T_P2 + 1e-9 + EDGE, VDD * x),
                 (T_P3 - 2e-9, VDD * x), (T_P3 - 2e-9 + EDGE, 0), (T_END, 0)]
        lines.append(f"VWL{i} wl{i} 0 {pwl(shifted(w_pts))}")
        # FeFET: drain/source naming per SPICE: d=sl (VDD), g=wl, s=x
        lines.append(f"MF{i} sl wl{i} x{i} 0 nmos W={FEFET_W} L={FEFET_L} delvto={d:.5f} "
                     f"AD={ad_fe} AS={ad_fe} PD={pd_fe} PS={pd_fe}")
        lines.append(f"CM{i} x{i} 0 {C_MOM}")
        bl = f"bl{i}" if with_wire else "bl0"
        lines.append(f"MSN{i} x{i} s {bl} 0 nmos W={SW_W} L={SW_L} "
                     f"AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}")
        lines.append(f"MSP{i} x{i} sb {bl} sl pmos W={SW_W} L={SW_L} "
                     f"AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}")

    if with_wire:
        for i in range(n):
            lines.append(f"CW{i} bl{i} 0 {ROW_C}")
            if i + 1 < n:
                lines.append(f"RW{i} bl{i} bl{i + 1} {ROW_R}")

    lines.append(f"MR bl0 rst 0 0 nmos W={RST_W} L={RST_L} "
                 f"AD={RST_W * CONTACT} AS={RST_W * CONTACT} "
                 f"PD={2 * (RST_W + CONTACT)} PS={2 * (RST_W + CONTACT)}")
    return lines


def column_netlist(vths, inputs, with_wire=True) -> str:
    n = len(vths)
    lines = [f"* 1F1C column, N={n}", f".include {MODEL}"] + column_circuit(vths, inputs, with_wire, 0.0)
    lines += [
        f".tran 0.05n {T_END}",
        ".control",
        "run",
        f"meas tran vbl find v(bl0) at={T_END - 1e-9}",
        f"meas tran vbl_top find v(bl{n - 1 if with_wire else 0}) at={T_END - 1e-9}",
        f"let xavg = ({' + '.join(f'v(x{i})' for i in range(n))})/{n}",
        f"meas tran xavg_p2 find xavg at={T_P3 - 1e-9}",
        # energy from SL during P2 (cell charging) and during the whole cycle
        f"meas tran q_sl_p2 integ i(VSL) from={T_P2} to={T_P3}",
        f"meas tran q_sl_all integ i(VSL) from=0 to={T_END}",
        # time for BL (bottom) to settle within 1 mV of its final value
        f"meas tran t_settle when v(bl0)={{vbl-1e-3}} rise=1 from={T_P3}",
        ".endc",
        ".end",
    ]
    return "\n".join(lines) + "\n"


def simulate(vths, inputs, name, with_wire=True):
    out = run_ngspice(column_netlist(vths, inputs, with_wire), name)
    res = {"vbl": meas(out, "vbl"), "vbl_top": meas(out, "vbl_top"), "xavg_p2": meas(out, "xavg_p2"),
           "e_sl_p2": -meas(out, "q_sl_p2") * VDD, "e_sl_all": -meas(out, "q_sl_all") * VDD}
    try:
        res["t_settle"] = meas(out, "t_settle") - T_P3
    except RuntimeError:
        res["t_settle"] = float("nan")
    return res


# ---------------------------------------------------------------- experiments
def kappa_full_scale(n=N_ROWS):
    """All cells at the lowest Vth state, all inputs 1 -> compare VBL to the cells' own
    pre-share voltage."""
    r = simulate([min(VTH_STATES)] * n, [1] * n, f"kappa_n{n}")
    k_sim = r["vbl"] / r["xavg_p2"]
    k_paper = n * C_MOM / (n * C_MOM + LALENI["cpar_fF"] * 1e-15)
    print(f"[kappa N={n}] cells before sharing = {r['xavg_p2']:.4f} V, VBL = {r['vbl']:.4f} V "
          f"(top of BL {r['vbl_top']:.4f}), kappa_sim = {k_sim:.3f}, Laleni Eq.5 = {k_paper:.3f}, "
          f"BL settle (1 mV) = {r['t_settle'] * 1e9:.2f} ns")
    save_result("column", "kappa", {
        "rows": n, "vth_V": min(VTH_STATES),
        "cell_voltage_before_sharing_V": r["xavg_p2"], "vbl_full_scale_V": r["vbl"],
        "kappa": k_sim, "kappa_eq5_with_laleni_cpar": k_paper, "laleni_kappa": LALENI["kappa"],
        "bl_settle_1mV_ns": r["t_settle"] * 1e9})
    return r


def bl_charging_energy(n=N_ROWS):
    """Laleni: 'BL charging power is 0.86 nW with the expected value of weight as (VDD/2)'
    -> every cell programmed so its drain charges to VDD/2 (Vth = VDD/2), all inputs 1."""
    r = simulate([VDD / 2] * n, [1] * n, f"energy_n{n}")
    e_cell = r["e_sl_p2"] / n
    f = 1 / T_CLK
    print(f"[energy N={n}, Vd~VDD/2] cells reach {r['xavg_p2']:.4f} V; SL energy in P2 = "
          f"{r['e_sl_p2'] * 1e15:.2f} fJ per column = {e_cell * 1e15:.3f} fJ per cell; "
          f"x {f / 1e6:.0f} MHz = {e_cell * f * 1e9:.2f} nW per cell, {r['e_sl_p2'] * f * 1e9:.1f} nW per column "
          f"(Laleni: {LALENI['bl_charging_power_nW']} nW)")
    save_result("column", "bl_charging", {
        "rows": n, "vth_V": VDD / 2, "cell_voltage_V": r["xavg_p2"],
        "energy_per_column_per_mac_fJ": r["e_sl_p2"] * 1e15,
        "energy_per_cell_per_mac_fJ": e_cell * 1e15,
        "power_per_cell_at_clock_nW": e_cell * f * 1e9,
        "laleni_bl_charging_power_nW": LALENI["bl_charging_power_nW"]})
    return r


def fig10(vth_set, label):
    """Laleni Fig. 10: column of 4 FeFETs, all inputs 1, all 4^4 weight combinations,
    VBL without Cpar (no bitline wire)."""
    weight_of = {v: w for w, v in enumerate(sorted(vth_set, reverse=True))}  # highest Vth = 0
    points = []
    for combo in itertools.product(vth_set, repeat=4):
        mac = sum(weight_of[v] for v in combo)
        r = simulate(list(combo), [1] * 4, "fig10", with_wire=False)
        points.append((mac, r["vbl"]))
    by_mac = {}
    for mac, v in points:
        by_mac.setdefault(mac, []).append(v)
    print(f"[Fig. 10{label}] Vth = {vth_set}")
    for mac in sorted(by_mac):
        vs = by_mac[mac]
        print(f"   MAC={mac:2d}: VBL mean {sum(vs) / len(vs):.4f} V  (min {min(vs):.4f}, max {max(vs):.4f}, n={len(vs)})")
    save_result("column", f"fig10{label.strip('()')}", {
        "vth_states_V": list(vth_set), "rows": 4, "bitline_wire": False,
        "vbl_by_mac_V": {str(m): {"mean": sum(v) / len(v), "min": min(v), "max": max(v), "n": len(v)}
                          for m, v in sorted(by_mac.items())},
        "laleni_eq3_max_V": VDD - min(vth_set)})
    return by_mac


def cell_charging():
    """One cell per Vth state, no bitline wire: drain voltage at the end of the multiply
    phase vs Laleni eq. 1 (VDD - Vth), and VBL after sharing with the cell's own
    switch/reset parasitics. Splits the Fig. 10(b) gap into charging vs sharing loss."""
    states = sorted(set(VTH_STATES) | set(VTH_STATES_LINEAR))
    out = {}
    for v in states:
        r = simulate([v], [1], "cell_charging", with_wire=False)
        out[str(v)] = {"drain_V": r["xavg_p2"], "eq1_V": max(0.0, VDD - v), "vbl_after_sharing_V": r["vbl"]}
        print(f"[cell] Vth={v}: drain {r['xavg_p2']:.4f} V (eq. 1: {max(0.0, VDD - v):.2f}), "
              f"after sharing {r['vbl']:.4f} V")
    save_result("column", "cell_charging", {
        "multiply_phase_ns": (T_P3 - T_P2) * 1e9, "by_vth_V": out})
    return out


def fet_to_cap_load(t_max=100e-6, points_per_decade=8):
    """FeFET -> CMoM load (the multiply phase): stored level vs wordline pulse width.

    One cell as in the column: FeFET from SL (= VDD), drain on X with CMoM, bitline held
    at 0 by the reset device. The switch is on for the first 5 ns (discharging X, as in
    the reset phase), then off (its drain junction stays on X). The wordline is pulsed to
    VDD for a width tw, then returned to 0; V(X) is read 20 ns after it falls, i.e. the
    charge actually stored (while WL is high, the FeFET gate capacitance couples WL onto
    X, so V(X) during the pulse is not the stored value).

    Finding: the stored level has no final value. Near VDD - Vth the FeFET is in
    subthreshold and keeps charging the 1.3 fF CMoM, ~logarithmically in time, for as
    long as WL is high (Laleni's Fig. 6 flattens only because of its 10 pF load). So the
    load time is not a settling time but a design choice: the WL pulse width, which sets
    the levels together with Vth. Recorded here, for every Vth state:
      - stored level vs pulse width (5 ns .. t_max)
      - level at the design's multiply phase, and the steps between adjacent states
      - slope dV/dlog10(tw) at the multiply phase (sensitivity to pulse-width error)
      - off-state (highest Vth) stored level vs width: charge leaking into a '0' weight"""
    import math
    lsb = DESIGN["adc"]["vrefp_V"] / 2 ** DESIGN["adc"]["bits"]
    ad_fe, pd_fe = FEFET_W * CONTACT, 2 * (FEFET_W + CONTACT)
    ad_sw, pd_sw = SW_W * CONTACT, 2 * (SW_W + CONTACT)
    # reset: switch on (bitline at 0) until t_rst, discharging X, then off before WL rises
    t_rst, t_on, t_hold = 5e-9, 6e-9, 20e-9
    n_dec = math.log10(t_max / 5e-9)
    widths = [5e-9 * 10 ** (i / points_per_decade) for i in range(int(round(n_dec * points_per_decade)) + 1)]
    widths[-1] = t_max
    if (T_P3 - T_P2) not in widths:
        widths = sorted(widths + [T_P3 - T_P2])

    def stored(d, tw):
        t_off = t_on + tw
        net = "\n".join([
            "* FeFET -> CMoM, one cell, WL pulse", f".include {MODEL}",
            f"VSL sl 0 {VDD}",
            f"VWL wl 0 {pwl([(0, 0), (t_on, 0), (t_on + EDGE, VDD), (t_off, VDD), (t_off + EDGE, 0), (t_off + t_hold, 0)])}",
            f"VS s 0 {pwl([(0, VDD), (t_rst, VDD), (t_rst + EDGE, 0), (t_off + t_hold, 0)])}",
            f"VSB sb 0 {pwl([(0, 0), (t_rst, 0), (t_rst + EDGE, VDD), (t_off + t_hold, VDD)])}",
            f"MF sl wl x 0 nmos W={FEFET_W} L={FEFET_L} delvto={d:.5f} AD={ad_fe} AS={ad_fe} PD={pd_fe} PS={pd_fe}",
            f"CM x 0 {C_MOM}",
            f"MSN x s bl 0 nmos W={SW_W} L={SW_L} AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}",
            f"MSP x sb bl sl pmos W={SW_W} L={SW_L} AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}",
            "VBL bl 0 0", ".save v(x)",
            f".tran 0.05n {t_off + t_hold:.10e}",
            ".control", "run", f"meas tran vx find v(x) at={t_off + t_hold - 1e-10:.10e}", ".endc", ".end"]) + "\n"
        return meas(run_ngspice(net, "fet_to_cap"), "vx")

    states = sorted(set(VTH_STATES) | set(VTH_STATES_LINEAR))
    t_ph = T_P3 - T_P2
    i_ph = widths.index(t_ph)
    out = {}
    for v in states:
        d = delvto_for(v)
        vs = [stored(d, tw) for tw in widths]
        slope = (vs[i_ph + 1] - vs[i_ph - 1]) / math.log10(widths[i_ph + 1] / widths[i_ph - 1])
        out[str(v)] = {"stored_at_phase_V": vs[i_ph], "eq1_V": max(0.0, VDD - v),
                       "slope_V_per_decade_at_phase": slope,
                       "stored_by_width_ns": {f"{tw * 1e9:.4g}": x for tw, x in zip(widths, vs)}}
        print(f"[fet->cap] Vth={v}: {vs[i_ph]:.4f} V after {t_ph * 1e9:.0f} ns (eq. 1 {max(0.0, VDD - v):.2f}), "
              f"slope {slope * 1e3:.1f} mV/decade; 5 ns {vs[0]:.4f}, {t_max * 1e6:.0f} us {vs[-1]:.4f} V")

    def steps(vth_set):
        lv = [out[str(v)]["stored_at_phase_V"] for v in sorted(vth_set)]   # low Vth = high level
        return [lv[i] - lv[i + 1] for i in range(len(lv) - 1)]

    off = str(max(states))
    data = {
        "multiply_phase_ns": t_ph * 1e9, "lsb_mV": lsb * 1e3,
        "by_vth_V": out,
        "steps_at_phase_V": {"vth_states": steps(VTH_STATES), "vth_states_linear": steps(VTH_STATES_LINEAR)},
        "off_state_leak_V_by_width_ns": out[off]["stored_by_width_ns"],
        "pulse_width_1pct_error_mV": max(abs(r["slope_V_per_decade_at_phase"]) for r in out.values())
                                      * math.log10(1.01) * 1e3,
        "conclusion": "no settling: stored level rises ~logarithmically while WL is high; the load "
                      "time is the WL pulse width (design choice), one clock cycle in the design"}
    save_result("column", "fet_to_cap", data)
    print(f"steps at {t_ph * 1e9:.0f} ns: {data['steps_at_phase_V']}; 1% pulse-width error -> "
          f"{data['pulse_width_1pct_error_mV']:.2f} mV (LSB {lsb * 1e3:.2f} mV)")
    return out


def fig6(vth=0.3, vs_list=(0.10, 0.30, 0.50, 0.75, 1.00, 1.25, 1.50)):
    """Laleni Fig. 6 (measured): one LVT FeFET (Vth ~0.3 V, W/L 450/450 nm) as a switch,
    source at Vs, WL pulsed to 1 V for 100 us, drain loaded by ~10 pF (measurement setup).
    Reports the drain voltage at the end of the WL pulse."""
    d = delvto_for(vth)
    ad, pd = FEFET_W * CONTACT, 2 * (FEFET_W + CONTACT)
    t_on, t_off, edge = 100e-6, 200e-6, 5e-6
    print(f"[Fig. 6] Vth={vth} V (delvto {d:.4f}), WL=1 V for 100 us, 10 pF load")
    res = {}
    for vs in vs_list:
        net = f"""* Laleni Fig. 6
.include {MODEL}
VSL sl 0 {vs}
VWL wl 0 {pwl([(0, 0), (t_on, 0), (t_on + edge, VDD), (t_off, VDD), (t_off + edge, 0), (t_off + 50e-6, 0)])}
MF sl wl x 0 nmos W={FEFET_W} L={FEFET_L} delvto={d:.5f} AD={ad} AS={ad} PD={pd} PS={pd}
CL x 0 10p
.tran 0.1u {t_off + 50e-6}
.control
run
meas tran vd find v(x) at={t_off - 1e-9}
.endc
.end
"""
        res[vs] = meas(run_ngspice(net, "fig6"), "vd")
        print(f"   Vs={vs:.2f} V: Vd = {res[vs]:.4f} V   (min(Vs, VWL-Vth) = {min(vs, VDD - vth):.2f})")
    save_result("column", "fig6", {
        "vth_V": vth, "wl_V": VDD, "pulse_us": (t_off - t_on) * 1e6, "load_pF": 10,
        "drain_V_by_vs": {f"{vs:.2f}": res[vs] for vs in vs_list},
        "measured_plateau_V_read_off_fig6": 0.70, "ideal_min_vs_vgs_minus_vth_V":
            {f"{vs:.2f}": min(vs, VDD - vth) for vs in vs_list}})
    return res


if __name__ == "__main__":
    fig6()
    cell_charging()
    fet_to_cap_load()
    kappa_full_scale()
    bl_charging_energy()
    fig10(VTH_STATES, "(a)")
    fig10(VTH_STATES_LINEAR, "(b)")
