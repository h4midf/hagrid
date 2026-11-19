"""Write one value into a cell's CMoM and read it back, in SPICE, on the full
128-cell column (bitline wire from NVSim) driving the SAR ADC.

This is our extension to Laleni's macro (it has no capacitor write/read path):
  write: a per-column 4-level driver -- four shared reference voltages, one selected
         through an LVT transmission gate onto the bitline -- charges the target row's
         CMoM through that row's switch S (all other rows' S off).
  read : bitline reset to 0, then two ADC conversions with the ADC reference lowered
         to VREF_READ: (1) bitline alone (reference), (2) after closing the row's S
         (the cell's charge shared onto the bitline). value = code2 - code1, which
         cancels the comparator/buffer offset and the switch charge injection.
         The read is destructive (the cell's charge is spread over the bitline).

Write levels = the four levels the FeFET load stores at the multiply phase
(results/column.json -> fet_to_cap, linear Vth states), so written temp data uses the
same encoding as the MAC.

Timeline (ADC conversion k occupies [k*T_CONV, (k+1)*T_CONV]; conversion 0 flushes):
  write  : [T_W0, T_W1] driver on + row S on; S off at T_W1, driver off after
  reset  : RST on [T_W1 + 10 ns, T_CONV - 1 ns]
  conv 1 : samples the bare bitline (reference), 52 ns after the reset releases
  reset  : RST on again [T_CONV + 60 ns, 2 T_CONV - 1 ns]
  share  : row S on at T_SHARE = the moment that reset releases
  conv 2 : samples the shared bitline (signal), also 52 ns after release
The floating bitline drifts up (~80 mV/us, leakage through the off switches); both
conversions sample at the same time after release, so the drift cancels in code2 - code1.

Writes results/mac.json ("cap_write", "cap_read").
Usage: python3 targets/1fefet1c/spice/mac/cap_rw.py
"""
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "column"))
sys.path.insert(0, str(HERE.parent / "adc"))
import column_sim as col  # noqa: E402
import run_mc  # noqa: E402
import tb as adc  # noqa: E402

VDD = col.VDD
N = col.N_ROWS
ROW = N - 1                    # target row: top of the bitline, farthest from the driver
IBUF = adc.DESIGN["adc"]["buffer_bias_uA"] * 1e-6   # buffer bias used for all ADC results (design.json)
VREF_READ = 0.064              # ADC reference in read mode: cell signal + bitline drift + offsets
DRIVER_W = 384e-9              # driver transmission gate width
T_CONV = adc.T_CONV
T_W0, T_W1 = 100e-9, 150e-9    # write window
T_SHARE = 2 * T_CONV - 1e-9    # close the row's S as the second reset releases
T_CODE1, T_CODE2 = 2 * T_CONV + 40e-9, 3 * T_CONV + 40e-9
T_END = 3 * T_CONV + 50e-9
WRITE_TOL = 0.01               # write settled when within 1 % of the level
WRITE_LATENCY = T_W1 - T_W0    # the clocked write window (the write settles inside it: t_write_1pct_ns)
READ_LATENCY = 3 * T_CONV - (T_CONV - 1e-9)   # reset release before conv 1 -> end of conv 2


def levels():
    """The four stored levels of the FeFET load at the multiply phase (linear states)."""
    by = col.load_result("column")["fet_to_cap"]["by_vth_V"]
    return [max(0.0, by[str(v)]["stored_at_phase_V"]) for v in sorted(col.VTH_STATES_LINEAR, reverse=True)]


def array_lines(level_idx: int, lv: list[float]) -> list[str]:
    ad_fe, pd_fe = col.FEFET_W * col.CONTACT, 2 * (col.FEFET_W + col.CONTACT)
    ad_sw, pd_sw = col.SW_W * col.CONTACT, 2 * (col.SW_W + col.CONTACT)
    e = col.EDGE
    off = col.delvto_for(max(col.VTH_STATES))     # cells hold the off state; WL stays low
    lines = [f"VSL sl 0 {VDD}", "VWLOFF wloff 0 0", f"VHI hi 0 {VDD}"]
    # row switch control: target row on during the write and from T_SHARE; others always off
    s_pts = [(0, 0), (T_W0, 0), (T_W0 + e, VDD), (T_W1, VDD), (T_W1 + e, 0),
             (T_SHARE, 0), (T_SHARE + e, VDD), (T_END, VDD)]
    lines.append(f"VSR sr 0 {col.pwl(s_pts)}")
    lines.append(f"VSRB srb 0 {col.pwl([(t, VDD - v) for t, v in s_pts])}")
    # bitline reset: on before the write, off during it, on after it until conversion 1
    r_pts = [(0, VDD), (T_W0 - 5e-9, VDD), (T_W0 - 5e-9 + e, 0), (T_W1 + 10e-9, 0),
             (T_W1 + 10e-9 + e, VDD), (T_CONV - 1e-9, VDD), (T_CONV - 1e-9 + e, 0),
             (T_CONV + 60e-9, 0), (T_CONV + 60e-9 + e, VDD), (T_SHARE, VDD), (T_SHARE + e, 0), (T_END, 0)]
    lines.append(f"VRST rst 0 {col.pwl(r_pts)}")
    for i in range(N):
        s, sb = ("sr", "srb") if i == ROW else ("0", "hi")
        lines += [
            f"MF{i} sl wloff x{i} 0 nmos W={col.FEFET_W} L={col.FEFET_L} delvto={off:.5f} "
            f"AD={ad_fe} AS={ad_fe} PD={pd_fe} PS={pd_fe}",
            f"CM{i} x{i} 0 {col.C_MOM}",
            f"MSN{i} x{i} {s} bl{i} 0 nmos W={col.SW_W} L={col.SW_L} AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}",
            f"MSP{i} x{i} {sb} bl{i} sl pmos W={col.SW_W} L={col.SW_L} AD={ad_sw} AS={ad_sw} PD={pd_sw} PS={pd_sw}",
            f"CW{i} bl{i} 0 {col.ROW_C}"]
        if i + 1 < N:
            lines.append(f"RW{i} bl{i} bl{i + 1} {col.ROW_R}")
    lines.append(f"MR bl0 rst 0 0 nmos W={col.RST_W} L={col.RST_L} AD={col.RST_W * col.CONTACT} "
                 f"AS={col.RST_W * col.CONTACT} PD={2 * (col.RST_W + col.CONTACT)} PS={2 * (col.RST_W + col.CONTACT)}")
    # 4-level driver: shared references, the selected one through an LVT TG onto bl0
    d_pts = [(0, 0), (T_W0, 0), (T_W0 + e, VDD), (T_W1 + 2e-9, VDD), (T_W1 + 2e-9 + e, 0), (T_END, 0)]
    lines.append(f"VDEN den 0 {col.pwl(d_pts)}")
    lines.append(f"VDENB denb 0 {col.pwl([(t, VDD - v) for t, v in d_pts])}")
    for k, v in enumerate(lv):
        lines.append(f"VW{k} wref{k} 0 {v:.6f}")
        en, enb = ("den", "denb") if k == level_idx else ("0", "hi")
        lines.append(f"XD{k} wref{k} bl0 {en} {enb} vdd tglvt W={DRIVER_W}")
    lines.append("VLINK bl0 bl 0")
    return lines


def netlist(level_idx, lv, cmp_offset):
    net = adc.testbench(adc.FULL_TB).replace(".param ibuf=1u", f".param ibuf={IBUF}")
    net = net.replace("VREF vref 0 {vref_v}", f"VREF vref 0 {VREF_READ}")
    net = net.replace("VBL  bl   0 PWL(__PWL__)", "\n".join(array_lines(level_idx, lv)))
    if cmp_offset:
        dvt = {d[0]: 0.0 for d in run_mc.CMP_DEVICES}
        dvt["M1"] = cmp_offset
        net = net.replace(".param ibuf=", run_mc.comparator_mc(dvt) + "\n.param ibuf=", 1)
        net = net.replace("Xcmp top vcm clkc cmp vdd comparator", "Xcmp top vcm clkc cmp vdd comparator_mc")
    target = lv[level_idx]
    meas = [
        f"meas tran v_written find v(x{ROW}) at={T_W1 + 5e-9:.6e}",
        f"meas tran v_held find v(x{ROW}) at={T_SHARE - 1e-9:.6e}",
        f"meas tran v_neighbor0 find v(x{ROW - 1}) at={T_W0 - 1e-9:.6e}",
        f"meas tran v_neighbor find v(x{ROW - 1}) at={T_SHARE - 1e-9:.6e}",
        f"meas tran q_write integ i(VW{level_idx}) from={T_W0:.6e} to={T_W1 + 2e-9:.6e}",
        f"meas tran vbl_ref find v(bl) at={T_CONV + 50e-9:.6e}",
        f"meas tran vbl_sig find v(bl) at={2 * T_CONV + 50e-9:.6e}",
        f"meas tran code1 find code at={T_CODE1:.6e}",
        f"meas tran code2 find code at={T_CODE2:.6e}",
    ]
    if target > 0:
        meas.append(f"meas tran t_write when v(x{ROW})={target * (1 - WRITE_TOL):.6e} rise=1 from={T_W0:.6e}")
    net = net.replace("__TEND__", f"{T_END:.6e}").replace("__MEAS__", "\n".join(meas))
    return net.replace(".control\nrun", f".save v(bl) v(x{ROW}) v(x{ROW - 1}) i(VW{level_idx}) "
                       f"{' '.join(adc.BIT_NODES)}\n.control\nrun")


def run(args):
    level_idx, lv, cmp_offset = args
    out = adc.run_ngspice(netlist(level_idx, lv, cmp_offset), f"caprw_L{level_idx}_off{cmp_offset * 1e3:.0f}mV")
    r = {"level": level_idx, "target_V": lv[level_idx], "comparator_offset_mV": cmp_offset * 1e3}
    for k in ("v_written", "v_held", "v_neighbor0", "v_neighbor", "vbl_ref", "vbl_sig", "code1", "code2"):
        r[k] = adc.meas(out, k)
    r["neighbor_disturb_mV"] = (r["v_neighbor"] - r["v_neighbor0"]) * 1e3
    r["write_energy_from_reference_fJ"] = -adc.meas(out, "q_write") * lv[level_idx] * 1e15
    tw = adc.meas(out, "t_write") if lv[level_idx] > 0 else float("nan")
    r["t_write_1pct_ns"] = (tw - T_W0) * 1e9 if tw == tw else None
    r["read_value_codes"] = r["code2"] - r["code1"]
    r["signal_mV"] = (r["vbl_sig"] - r["vbl_ref"]) * 1e3
    return r


def main():
    lv = levels()
    col.delvto_for(max(col.VTH_STATES))   # calibrate once before the parallel runs
    jobs = [(k, lv, off) for off in (0.0, 15e-3) for k in range(len(lv))]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(run, jobs))
    lsb_read = VREF_READ / 2 ** adc.BITS
    print(f"levels {[round(x, 4) for x in lv]} V; read-mode LSB {lsb_read * 1e3:.3f} mV")
    print(f"{'lvl':>3} {'cmp off':>7} {'written':>8} {'held':>7} {'t_wr ns':>8} {'E_wr fJ':>8} "
          f"{'sig mV':>7} {'code1':>5} {'code2':>5} {'value':>5}")
    for r in res:
        tw = "-" if r["t_write_1pct_ns"] is None else f"{r['t_write_1pct_ns']:.2f}"
        print(f"{r['level']:>3} {r['comparator_offset_mV']:>7.0f} {r['v_written']:8.4f} {r['v_held']:7.4f} {tw:>8} "
              f"{r['write_energy_from_reference_fJ']:8.1f} {r['signal_mV']:7.2f} {r['code1']:5.0f} "
              f"{r['code2']:5.0f} {r['read_value_codes']:5.0f}")
    adc.save_result("mac", "cap_write", {
        "latency_ns": WRITE_LATENCY * 1e9,
        "row": ROW, "driver_tg_width_nm": DRIVER_W * 1e9, "write_window_ns": (T_W1 - T_W0) * 1e9,
        "tolerance_pct": WRITE_TOL * 100, "levels_V": lv,
        "by_level": [{k: r[k] for k in ("level", "target_V", "v_written", "v_held", "neighbor_disturb_mV",
                                        "t_write_1pct_ns", "write_energy_from_reference_fJ")}
                     for r in res if r["comparator_offset_mV"] == 0]})
    path = adc.save_result("mac", "cap_read", {
        "latency_ns": READ_LATENCY * 1e9,
        "row": ROW, "vref_read_V": VREF_READ, "lsb_read_mV": lsb_read * 1e3, "conversions": 2,
        "by_case": [{k: r[k] for k in ("level", "target_V", "comparator_offset_mV", "vbl_ref", "vbl_sig",
                                       "signal_mV", "code1", "code2", "read_value_codes")} for r in res]})
    print("->", path)


if __name__ == "__main__":
    main()
