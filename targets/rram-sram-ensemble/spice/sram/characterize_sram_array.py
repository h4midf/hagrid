"""The SRAM macro's 32 x 64 array at transistor level, on the chip's low-power devices
(spice/logic/inverter.py's fitted model), replacing CACTI's power. Writes
default/results/sram_array_spice.json.

One read drives every column: the word line opens all 64 cells for word_line_pulse_ns, one
bitline of every pair swings fully and is precharged back; 16 of the columns go through the
4-to-1 mux to a StrongARM sense amplifier. So, per read:

  energy = 16 x selected column + 48 x unselected column + word line + precharge line
           + sense clock line + column select line + 0.5 x row predecode line

One write of a 16-bit word: write drivers behind the 4-to-1 mux pull one bitline of each
of the 16 selected columns low while the word line is open, and the cell must flip; the
other 48 columns are half-selected and swing as in a read:

  energy = 16 x written column + 48 x unselected column + word line + precharge line
           + column select line + 0.5 x row predecode line + write-enable line

A column simulates its active 6T cell, the other 31 cells' access transistors as load and
its wires. The bitline must swing nearly fully and the selected column's sense amplifier
must read the stored value. Addresses step by one, so the column select
changes every read and the row every 4 reads (2 predecode lines per row change).

  power = energy x read rate + leakage of the 2048 cells

Assumptions: design.json "model_assumptions.sram_array".
Usage (from the repo root): python3 targets/rram-sram-ensemble/spice/sram/characterize_sram_array.py
"""
import json
import os
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..", "..")
sys.path[:0] = [os.path.join(HERE, ".."), os.path.join(HERE, "..", "logic")]
from inverter import model_text  # noqa: E402
from thick_model import DESIGN, get  # noqa: E402

A = DESIGN["model_assumptions"]["sram_array"]
LAYOUT = DESIGN["model_assumptions"]["array_layout"]
VDD = DESIGN["model_assumptions"]["readout"]["vdd_V"]
SM = DESIGN["sram_macro"]
ROWS, COLS = SM["array"]["rows"], SM["array"]["cols"]
SETS, MUX = SM["column_sets"], SM["column_mux"]
TCK_NS = 1e3 / DESIGN["clock"]["freq_MHz"]
LOGIC = json.load(open(os.path.join(TARGET, "default", "results", "logic.json")))
FLASH_CELLS = os.path.abspath(os.path.join(HERE, "..", "readout", "flash_cells.lib"))
WIRE = LAYOUT["wire_fF_per_um"]
MUX_N_NM, MUX_P_NM = 260, 520


def run(netlist: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "m.pm")
        with open(model, "w") as f:
            f.write(model_text(LOGIC["vth_n_V"], LOGIC["vth_p_V"]))
        path = os.path.join(tmp, "n.sp")
        with open(path, "w") as f:
            f.write(netlist.replace("__MODEL__", model))
        return subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout


def column(selected: bool, pw_ns: float) -> str:
    """Netlist of one column over two reads, with a word-line pulse of pw_ns."""
    c_bl = ROWS * A["cell_height_um"] * WIRE
    loads = "\n".join(f"MB{k} bl 0 {'vc' if k % 2 else '0'} 0 nmos w={A['access_w_nm']}n l=65n\n"
                      f"MC{k} blb 0 {'0' if k % 2 else 'vc'} 0 nmos w={A['access_w_nm']}n l=65n"
                      for k in range(ROWS - 1))
    sense = ""
    if selected:
        sense = f"""MNA bl vc sa 0 nmos w={MUX_N_NM}n l=65n
MPA bl 0 sa vc pmos w={MUX_P_NM}n l=65n
MNB blb vc sb 0 nmos w={MUX_N_NM}n l=65n
MPB blb 0 sb vc pmos w={MUX_P_NM}n l=65n
XS sa sb sck on op vsa sa_n
"""
    t_wl, t_sa = 0.3, 0.3 + pw_ns
    half = TCK_NS / 2
    return f""".include __MODEL__
.include {FLASH_CELLS}
VC vc 0 {VDD}
VPC vpc 0 {VDD}
VSA vsa 0 {VDD}
VWL wl 0 PULSE(0 {VDD} {t_wl}n 30p 30p {pw_ns}n {TCK_NS}n)
VP pc 0 PULSE(0 {VDD} 0.1n 30p 30p {half}n {TCK_NS}n)
VK sck 0 PULSE(0 {VDD} {t_sa}n 30p 30p 1n {TCK_NS}n)
MPD q qb 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MPU q qb vc vc pmos w={A['pull_up_w_nm']}n l=65n
MQD qb q 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MQU qb q vc vc pmos w={A['pull_up_w_nm']}n l=65n
MAX1 bl wl q 0 nmos w={A['access_w_nm']}n l=65n
MAX2 blb wl qb 0 nmos w={A['access_w_nm']}n l=65n
{loads}
CBL bl 0 {c_bl}f
CBB blb 0 {c_bl}f
MR1 bl pc vpc vpc pmos w={A['precharge_w_nm']}n l=65n
MR2 blb pc vpc vpc pmos w={A['precharge_w_nm']}n l=65n
MR3 bl pc blb vpc pmos w={A['precharge_w_nm']}n l=65n
{sense}
.ic v(q)=0 v(qb)={VDD} v(bl)={VDD} v(blb)={VDD}
.tran 5p {3 * TCK_NS}n uic
.control
run
meas tran dv_a find v(bl) at={TCK_NS + t_sa}n
meas tran dv_b find v(blb) at={TCK_NS + t_sa}n
meas tran q_vpc integ i(vpc) from={TCK_NS}n to={2 * TCK_NS}n
meas tran q_vc integ i(vc) from={TCK_NS}n to={2 * TCK_NS}n
meas tran q_vsa integ i(vsa) from={TCK_NS}n to={2 * TCK_NS}n
meas tran v_on find v({'on' if selected else 'vc'}) at={TCK_NS + t_sa + 0.8}n
meas tran v_op find v({'op' if selected else 'vc'}) at={TCK_NS + t_sa + 0.8}n
.endc
.end
"""


def column_energy_J(selected: bool, pw_ns: float) -> float:
    out = run(column(selected, pw_ns))
    swing = get(out, "dv_b") - get(out, "dv_a")
    if swing < 0.9 * VDD:
        raise RuntimeError(f"the bitline swings only {swing:.3f} V in {pw_ns} ns")
    if selected and not get(out, "v_op") < 0.2 * VDD < 0.8 * VDD < get(out, "v_on"):
        raise RuntimeError(f"the sense amplifier misread: on {get(out, 'v_on'):.3f} V, op {get(out, 'v_op'):.3f} V")
    # the supply's sign is negative when it delivers charge
    return -(get(out, "q_vpc") + get(out, "q_vc") + get(out, "q_vsa")) * VDD


def write_column(pw_ns: float) -> str:
    """Netlist of one written column over three cycles: write drivers on the far side of
    the column mux pull blb low in cycle 1 (writing q = 1) and bl low in cycle 2 (writing
    q = 0), each while the word line is open, so the cell flips every cycle."""
    c_bl = ROWS * A["cell_height_um"] * WIRE
    loads = "\n".join(f"MB{k} bl 0 {'vc' if k % 2 else '0'} 0 nmos w={A['access_w_nm']}n l=65n\n"
                      f"MC{k} blb 0 {'0' if k % 2 else 'vc'} 0 nmos w={A['access_w_nm']}n l=65n"
                      for k in range(ROWS - 1))
    t_wl = 0.3
    half = TCK_NS / 2
    wd = A["write_driver_w_nm"]
    return f""".include __MODEL__
VC vc 0 {VDD}
VPC vpc 0 {VDD}
VWL wl 0 PULSE(0 {VDD} {t_wl}n 30p 30p {pw_ns}n {TCK_NS}n)
VP pc 0 PULSE(0 {VDD} 0.1n 30p 30p {half}n {TCK_NS}n)
VWB wenb 0 PULSE(0 {VDD} {t_wl}n 30p 30p {pw_ns}n {2 * TCK_NS}n)
VWA wena 0 PULSE(0 {VDD} {TCK_NS + t_wl}n 30p 30p {pw_ns}n {2 * TCK_NS}n)
MPD q qb 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MPU q qb vc vc pmos w={A['pull_up_w_nm']}n l=65n
MQD qb q 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MQU qb q vc vc pmos w={A['pull_up_w_nm']}n l=65n
MAX1 bl wl q 0 nmos w={A['access_w_nm']}n l=65n
MAX2 blb wl qb 0 nmos w={A['access_w_nm']}n l=65n
{loads}
CBL bl 0 {c_bl}f
CBB blb 0 {c_bl}f
MR1 bl pc vpc vpc pmos w={A['precharge_w_nm']}n l=65n
MR2 blb pc vpc vpc pmos w={A['precharge_w_nm']}n l=65n
MR3 bl pc blb vpc pmos w={A['precharge_w_nm']}n l=65n
MNA bl vc sa 0 nmos w={MUX_N_NM}n l=65n
MPA bl 0 sa vc pmos w={MUX_P_NM}n l=65n
MNB blb vc sb 0 nmos w={MUX_N_NM}n l=65n
MPB blb 0 sb vc pmos w={MUX_P_NM}n l=65n
MWA sa wena 0 0 nmos w={wd}n l=65n
MWB sb wenb 0 0 nmos w={wd}n l=65n
.ic v(q)=0 v(qb)={VDD} v(bl)={VDD} v(blb)={VDD}
.tran 5p {3 * TCK_NS}n uic
.control
run
meas tran q_one find v(q) at={t_wl + pw_ns + 0.5}n
meas tran q_zero find v(q) at={TCK_NS + t_wl + pw_ns + 0.5}n
meas tran q_vpc integ i(vpc) from={TCK_NS}n to={2 * TCK_NS}n
meas tran q_vc integ i(vc) from={TCK_NS}n to={2 * TCK_NS}n
.endc
.end
"""


def write_column_energy_J(pw_ns: float) -> float:
    """Energy of one written column per write (cycle 2), checking that both writes flip the cell."""
    out = run(write_column(pw_ns))
    if not (get(out, "q_one") > 0.8 * VDD and get(out, "q_zero") < 0.2 * VDD):
        raise RuntimeError(f"the write does not flip the cell: q {get(out, 'q_one'):.3f} V after writing 1, "
                           f"{get(out, 'q_zero'):.3f} V after writing 0")
    # the supply's sign is negative when it delivers charge
    return -(get(out, "q_vpc") + get(out, "q_vc")) * VDD


def line_energy_J(gates: list, wire_fF: float, driver_scale: int) -> float:
    """Energy of one rise and fall of a driven line loaded by `gates` ([(type, w_nm, count)])."""
    loads, i = [], 0
    for kind, w, count in gates:
        for _ in range(count):
            tie = "0" if kind == "nmos" else "vd"
            loads.append(f"ML{i} {tie} line {tie} {tie} {kind} w={w}n l=65n")
            i += 1
    body = "\n".join(loads)
    period = 2 * TCK_NS
    out = run(f""".include __MODEL__
VD vd 0 {VDD}
VI in 0 PULSE(0 {VDD} {TCK_NS}n 50p 50p {TCK_NS}n {period}n)
MP line in vd vd pmos w={260 * driver_scale}n l=65n
MN line in 0 0 nmos w={130 * driver_scale}n l=65n
{body}
CW line 0 {wire_fF}f
.tran 10p {2 * period}n
.control
run
meas tran q integ i(vd) from={period}n to={2 * period}n
.endc
.end
""")
    return -get(out, "q") * VDD


def cell_leak_mW() -> float:
    out = run(f""".include __MODEL__
VC vc 0 {VDD}
VB bl 0 {VDD}
MPD q qb 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MPU q qb vc vc pmos w={A['pull_up_w_nm']}n l=65n
MQD qb q 0 0 nmos w={A['pull_down_w_nm']}n l=65n
MQU qb q vc vc pmos w={A['pull_up_w_nm']}n l=65n
MAX1 bl 0 q 0 nmos w={A['access_w_nm']}n l=65n
MAX2 bl 0 qb 0 nmos w={A['access_w_nm']}n l=65n
.nodeset v(q)=0 v(qb)={VDD}
.control
op
print i(vc)
print i(vb)
.endc
.end
""")
    return (abs(get(out, "i\\(vc\\)")) + abs(get(out, "i\\(vb\\)"))) * VDD * ROWS * COLS * 1e3


def main() -> None:
    pw = A["word_line_pulse_ns"]
    selected, unselected = column_energy_J(True, pw), column_energy_J(False, pw)
    wl = line_energy_J([("nmos", A["access_w_nm"], 2 * COLS)], COLS * A["cell_width_um"] * WIRE, 8)
    precharge = line_energy_J([("pmos", A["precharge_w_nm"], 3 * COLS)], COLS * A["cell_width_um"] * WIRE, 16)
    sense_clock = line_energy_J([("nmos", 520, SETS), ("pmos", 260, 4 * SETS)], COLS * A["cell_width_um"] * WIRE, 8)
    col_select = line_energy_J([("nmos", MUX_N_NM, 2 * SETS), ("pmos", MUX_P_NM, 2 * SETS)],
                               COLS * A["cell_width_um"] * WIRE, 8)
    predecode = line_energy_J([("nmos", 260, 8), ("pmos", 260, 8)], ROWS * A["cell_height_um"] * WIRE, 2)

    parts_pJ = {"selected_columns": SETS * selected * 1e12,
                "unselected_columns": (COLS - SETS) * unselected * 1e12,
                "word_line": wl * 1e12, "precharge_line": precharge * 1e12, "sense_clock": sense_clock * 1e12,
                "column_select": col_select * 1e12, "row_predecode": 0.5 * predecode * 1e12}
    read_pJ = sum(parts_pJ.values())
    weight_bits = DESIGN["model_assumptions"]["sram_logic"]["weight_bits"]
    rate_MHz = DESIGN["clock"]["freq_MHz"] * weight_bits / (weight_bits + 1)
    leak = cell_leak_mW()

    # one write of a 16-bit word: its 16 columns written through the mux, the other 48
    # half-selected (word line open: they swing as in a read), the same lines as a read
    # without the sense clock, and the write-enable line to the 16 drivers
    written = write_column_energy_J(pw)
    write_enable = line_energy_J([("nmos", A["write_driver_w_nm"], SETS)], COLS * A["cell_width_um"] * WIRE, 8)
    write_parts_pJ = {"written_columns": SETS * written * 1e12,
                      "half_selected_columns": (COLS - SETS) * unselected * 1e12,
                      "word_line": wl * 1e12, "precharge_line": precharge * 1e12,
                      "column_select": col_select * 1e12, "row_predecode": 0.5 * predecode * 1e12,
                      "write_enable": write_enable * 1e12}
    write_pJ = sum(write_parts_pJ.values())
    out = {"word_line_pulse_ns": pw, "energy_pJ_per_read": parts_pJ, "read_pJ": read_pJ, "read_rate_MHz": rate_MHz,
           "leakage_mW": leak, "power_mW": read_pJ * rate_MHz * 1e-3 + leak,
           "energy_pJ_per_write": write_parts_pJ, "write_pJ": write_pJ}
    path = os.path.join(TARGET, "default", "results", "sram_array_spice.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"word-line pulse {pw:.3f} ns; bitlines swing fully; sense amplifier reads correctly")
    for k, v in sorted(parts_pJ.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v:8.4f} pJ")
    print(f"read {read_pJ:.4f} pJ; {out['power_mW']:.4f} mW at {rate_MHz:.0f} MHz reads (leakage {leak:.6f} mW)")
    print(f"write {write_pJ:.4f} pJ per 16-bit word (both writes flip the cell)")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
