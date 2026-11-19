"""MAC from the capacitors (WMU MAC), in SPICE, on the full 128-cell column (bitline
wire from NVSim) driving the SAR ADC.

This is our extension to Laleni's macro (it has no capacitor MAC path). The caps hold
written values (the four write levels of cap_rw.py, set as initial conditions: the
write itself is simulated in cap_rw.py). One input bit: the bitline is reset, then the
row switch S closes only on the rows whose input bit is 1; their charge shares onto
the bitline and the ADC converts it. The FeFETs stay off (WL low). The MAC is
destructive: the selected caps' charge is spread over the bitline.

Unlike the FeFET MAC (all 128 S close, unselected caps hold 0), only n rows connect,
so the bitline voltage is sum(C * V_i) / (C_bitline + n * C): the scale depends on n.

Two conversions, same timeline as the cap read in cap_rw.py, at the normal MAC ADC
reference:
  conv 1 : bare bitline (reference)
  conv 2 : after the selected rows' S close (signal)
code2 alone = a single-conversion MAC; code2 - code1 = with offset/drift cancellation.
Result: code1 is the same in every case (the ADC's fixed offset, as in the RMU MAC), and
the sharing settles within the sample cycle, so the WMU MAC needs one conversion
(CONVERSIONS); latency = from S closing to the end of that conversion.

Writes results/mac.json ("wmu_mac").
Usage: python3 targets/1fefet1c/spice/mac/wmu_mac.py
"""
import random
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "column"))
sys.path.insert(0, str(HERE.parent / "adc"))
sys.path.insert(0, str(HERE))
import cap_rw as rw  # noqa: E402
import column_sim as col  # noqa: E402
import tb as adc  # noqa: E402

VDD = col.VDD
N = col.N_ROWS
IBUF = adc.DESIGN["adc"]["buffer_bias_uA"] * 1e-6   # buffer bias used for all ADC results (design.json)
T_CONV = adc.T_CONV
T_SHARE = 2 * T_CONV - 1e-9    # selected rows' S close as the second reset releases
T_REF_SAMPLE = T_CONV + 50e-9  # bitline at the end of conv 1's sample cycle
T_SIG_SAMPLE = 2 * T_CONV + 50e-9
T_SIG_LATE = 2 * T_CONV + 250e-9   # same bitline 200 ns later: has the sharing settled?
T_CODE1, T_CODE2 = 2 * T_CONV + 40e-9, 3 * T_CONV + 40e-9
T_END = 3 * T_CONV + 50e-9
CONVERSIONS = 1                # see the docstring: the reference conversion is not needed
LATENCY = 3 * T_CONV - T_SHARE # S closes -> end of the signal conversion


def array_lines(stored: list[float], selected: list[int]) -> list[str]:
    ad_fe, pd_fe = col.FEFET_W * col.CONTACT, 2 * (col.FEFET_W + col.CONTACT)
    ad_sw, pd_sw = col.SW_W * col.CONTACT, 2 * (col.SW_W + col.CONTACT)
    e = col.EDGE
    off = col.delvto_for(max(col.VTH_STATES))     # FeFETs off; WL stays low
    lines = [f"VSL sl 0 {VDD}", "VWLOFF wloff 0 0", f"VHI hi 0 {VDD}"]
    # selected rows' switch: off until T_SHARE, then on
    s_pts = [(0, 0), (T_SHARE, 0), (T_SHARE + e, VDD), (T_END, VDD)]
    lines.append(f"VSR sr 0 {col.pwl(s_pts)}")
    lines.append(f"VSRB srb 0 {col.pwl([(t, VDD - v) for t, v in s_pts])}")
    # bitline reset: on until conv 1, on again until T_SHARE
    r_pts = [(0, VDD), (T_CONV - 1e-9, VDD), (T_CONV - 1e-9 + e, 0),
             (T_CONV + 60e-9, 0), (T_CONV + 60e-9 + e, VDD), (T_SHARE, VDD), (T_SHARE + e, 0), (T_END, 0)]
    lines.append(f"VRST rst 0 {col.pwl(r_pts)}")
    sel = set(selected)
    for i in range(N):
        s, sb = ("sr", "srb") if i in sel else ("0", "hi")
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
    lines.append("VLINK bl0 bl 0")
    # the written values (write simulated in cap_rw.py)
    lines.append(".ic " + " ".join(f"v(x{i})={v:.6f}" for i, v in enumerate(stored)))
    return lines


def netlist(stored, selected):
    net = adc.testbench(adc.FULL_TB).replace(".param ibuf=1u", f".param ibuf={IBUF}")
    net = net.replace("VBL  bl   0 PWL(__PWL__)", "\n".join(array_lines(stored, selected)))
    meas = [
        f"meas tran vbl_ref find v(bl) at={T_REF_SAMPLE:.6e}",
        f"meas tran vbl_sig find v(bl) at={T_SIG_SAMPLE:.6e}",
        f"meas tran vbl_late find v(bl) at={T_SIG_LATE:.6e}",
        f"meas tran code1 find code at={T_CODE1:.6e}",
        f"meas tran code2 find code at={T_CODE2:.6e}",
    ]
    net = net.replace("__TEND__", f"{T_END:.6e}").replace("__MEAS__", "\n".join(meas))
    return net.replace(".control\nrun", f".save v(bl) {' '.join(adc.BIT_NODES)}\n.control\nrun")


def cases(lv):
    """(name, stored level index per row, selected rows)."""
    rng = random.Random(1)
    levels_idx = [rng.randrange(len(lv)) for _ in range(N)]
    top = [len(lv) - 1] * N
    out = []
    for n in (1, N // 4, N):
        out.append((f"top_n{n}", top, list(range(N - n, N))))
    for n in (1, N // 16, N // 4, N // 2, N):
        out.append((f"random_n{n}", levels_idx, sorted(rng.sample(range(N), n))))
    return out


def run(args):
    name, idx, selected, lv = args
    stored = [lv[k] for k in idx]
    out = adc.run_ngspice(netlist(stored, selected), f"wmu_mac_{name}")
    r = {"name": name, "n_selected": len(selected),
         "sum_level_idx": sum(idx[i] for i in selected),
         "sum_stored_V": sum(stored[i] for i in selected)}
    for k in ("vbl_ref", "vbl_sig", "vbl_late", "code1", "code2"):
        r[k] = adc.meas(out, k)
    r["code_diff"] = r["code2"] - r["code1"]
    r["settle_mV"] = (r["vbl_late"] - r["vbl_sig"]) * 1e3
    # scale: bitline volts per volt of stored charge = C / (C_bitline + n C)
    r["vbl_per_stored_V"] = (r["vbl_sig"] - r["vbl_ref"]) / r["sum_stored_V"]
    return r


def main():
    lv = rw.levels()
    col.delvto_for(max(col.VTH_STATES))   # calibrate once before the parallel runs
    jobs = [(name, idx, sel, lv) for name, idx, sel in cases(lv)]
    with ThreadPoolExecutor(max_workers=len(jobs)) as ex:
        res = list(ex.map(run, jobs))
    print(f"levels {[round(x, 4) for x in lv]} V; LSB {adc.LSB * 1e3:.3f} mV")
    # fit 1 / scale = C_bitline / C + n (least squares on the intercept, slope fixed at 1)
    c_ratio = sum(1 / r["vbl_per_stored_V"] - r["n_selected"] for r in res) / len(res)
    print(f"{'case':>12} {'n':>4} {'sum idx':>7} {'sum V':>7} {'vbl ref':>8} {'vbl sig':>8} "
          f"{'settle mV':>9} {'code1':>5} {'code2':>5} {'diff':>5} {'scale':>7}")
    for r in res:
        print(f"{r['name']:>12} {r['n_selected']:>4} {r['sum_level_idx']:>7} {r['sum_stored_V']:7.3f} "
              f"{r['vbl_ref']:8.4f} {r['vbl_sig']:8.4f} {r['settle_mV']:9.2f} "
              f"{r['code1']:5.0f} {r['code2']:5.0f} {r['code_diff']:5.0f} {r['vbl_per_stored_V']:7.4f}")
    print(f"C_bitline / C = {c_ratio:.1f}; latency {LATENCY * 1e9:.0f} ns ({CONVERSIONS} conversion)")
    path = adc.save_result("mac", "wmu_mac", {
        "conversions": CONVERSIONS, "latency_ns": LATENCY * 1e9, "c_bitline_over_c_cell": c_ratio,
        "lsb_mV": adc.LSB * 1e3, "buffer_bias_uA": IBUF * 1e6, "levels_V": lv,
        "share_at_ns": T_SHARE * 1e9, "sample_at_ns": T_SIG_SAMPLE * 1e9, "cases": res})
    print("->", path)


if __name__ == "__main__":
    main()
