"""Energy per operation of every SFU function, from gate-level simulation of the mapped
NanGate45 netlists on realistic operands, and the SFU's leakage. Writes
results/sfu_power.json (45 nm and scaled to HyFlexPIM's 65 nm).

Method: each circuit is synthesized with the same flow as characterize_sfu.py
(synth.synth(sized=True)) and written as a mapped netlist; yosys simulates it on an
operand stream (read_liberty gives the cell functions), and every cell input pin's
toggles are weighted by its liberty capacitance:
  dynamic energy = sum over input pins of toggles x 0.5 C V^2   (V = 1.1 V)
the same switching model as custom_op_library/common/tools/report.py, with the
measured toggles in place of its fixed alpha = 0.2. Zero-delay simulation: no glitches,
no cell internal power, no wire capacitance (all only add). Clock pins are counted
separately: they toggle twice per cycle while a unit is clocked.

  unit functions (add, sub, mul, div, sqrt, max, to_int8): the unit alone, a new
    operand pair every cycle; energy / op = (logic energy + its clock energy) / ops
  exp: the whole lane in exp mode, one exp every 7 cycles (sfu_lane.v); the energy of
    the units exp does not use (their clock and their constant inputs) is removed

Operands (FP16, numpy): softmax rows of 128 attention scores ~ N(0, 2.5), layer norm
rows of 1024 hidden values ~ N(0, 1), GELU inputs ~ N(0, 1) (x / (1 + e^(-1.702 x)));
each function gets the operands it sees in those operations.

Usage (from the repo root):
  python3 targets/hyflexpim/sfu/power_sfu.py
"""
import json
import os
import re
import subprocess
import sys
import tempfile
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "..", "custom_op_library", "common", "tools")
sys.path.insert(0, TOOLS)
import synth  # noqa: E402
from liberty import LIB  # noqa: E402
from characterize_sfu import BASE, LANE, UNITS, factors  # noqa: E402

VDD, FREQ = 1.1, 1e9          # NanGate45 typical, and the SFU clock
N_OPS = 400                   # operations per unit run
N_EXP = 60                    # exps in the exp-mode run


# ---------------------------------------------------------------- netlist and simulation
def mapped_netlist(modules, top, path):
    """The mapped netlist of `top`, by the same yosys / ABC commands as synth.synth(sized=True)."""
    files = " ".join(os.path.join(HERE, "rtl", m + ".v") for m in modules)
    script = (f"read_verilog -sv {files}; hierarchy -top {top}; synth -flatten -top {top}; "
              f"dfflibmap -liberty {LIB}; abc -liberty {LIB} -script {synth._ABC_SIZED_SCRIPT}; opt_clean; "
              f"write_verilog -noattr {path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)


def simulate(mapped, tb_text, hexes, cycles, workdir):
    """VCD path of `cycles` cycles of the testbench (top tb) around the mapped netlist."""
    for name, values in hexes.items():
        with open(os.path.join(workdir, name), "w") as f:
            f.write("\n".join(f"{int(v) & 0xFFFF:04x}" for v in values))
    with open(os.path.join(workdir, "tb.v"), "w") as f:
        f.write(tb_text)
    script = (f"read_liberty -ignore_miss_func {LIB}; read_verilog {mapped}; read_verilog tb.v; "
              f"hierarchy -top tb; proc; memory; flatten; sim -clock clk -zinit -n {cycles} -vcd sim.vcd")
    subprocess.run(["yosys", "-q", "-p", script], cwd=workdir, check=True)
    return os.path.join(workdir, "sim.vcd")


# ---------------------------------------------------------------- pin capacitance and toggles
def pin_caps():
    caps, cell, pin, direction = {}, None, None, None
    with open(LIB) as f:
        for line in f:
            m = re.match(r'\s*cell\s*\(\s*"?(\w+)"?\s*\)', line)
            if m:
                cell = m.group(1)
                continue
            m = re.match(r'\s*pin\s*\(\s*"?(\w+)"?\s*\)', line)
            if m:
                pin, direction = m.group(1), None
                continue
            m = re.match(r'\s*direction\s*:\s*"?(\w+)"?', line)
            if m:
                direction = m.group(1)
            m = re.match(r"\s*capacitance\s*:\s*([\d.eE+-]+)", line)
            if m and direction == "input" and cell and pin and (cell, pin) not in caps:
                caps[(cell, pin)] = float(m.group(1))
    return caps


CAPS = pin_caps()


def instances(mapped):
    with open(mapped) as f:
        text = f.read()
    return {m.group(2).lstrip("\\"): m.group(1)
            for m in re.finditer(r"^\s*(\w+_X\d+)\s+(\S+)\s*\(", text, re.M)}


def energy_from_vcd(mapped, vcd, prefix="dut."):
    """(logic energy, clock-pin energy) in pJ over the whole simulation."""
    inst = instances(mapped)
    names, last, count = defaultdict(list), {}, defaultdict(int)
    t = 0
    with open(vcd) as f:
        for line in f:
            if line.startswith("$var"):
                p = line.split()
                if p[2] == "1" and p[4].startswith(prefix):
                    names[p[3]].append(p[4][len(prefix):])
            elif line.startswith("#"):
                t = int(line[1:])
            elif line.startswith("b") and " " in line:
                v, i = line[1:].split()
                if i in names:
                    if i in last and last[i] != v and t > 0:
                        count[i] += 1
                    last[i] = v
    logic = clock = 0.0
    for i, pins in names.items():
        for name in pins:
            if "." not in name:
                continue
            cell, pin = name.rsplit(".", 1)
            cap = CAPS.get((inst.get(cell), pin))
            if cap is None:
                continue
            e = 0.5 * cap * 1e-15 * VDD ** 2 * count[i] * 1e12
            if pin == "CK":
                clock += e
            else:
                logic += e
    return logic, clock


# ---------------------------------------------------------------- operands
def operands(seed=1):
    rng = np.random.default_rng(seed)
    h = lambda x: np.asarray(x, dtype=np.float16)                         # noqa: E731
    s = h(rng.normal(0, 2.5, (4, 128)))
    mx = s.max(1, keepdims=True)
    e = h(np.exp(s.astype(np.float32) - mx))
    sm = h(e.astype(np.float32).sum(1, keepdims=True))
    run_sum = h(np.cumsum(e, 1, dtype=np.float32))
    run_max = np.maximum.accumulate(s, 1)
    x = h(rng.normal(0, 1, (2, 1024)))
    mu = h(x.astype(np.float32).mean(1, keepdims=True))
    d = h(x - mu)
    var = h((d.astype(np.float32) ** 2).mean(1, keepdims=True))
    std = h(np.sqrt(var.astype(np.float32) + 1e-5))
    g = h(rng.normal(0, 1, 1024))
    ge = h(np.exp(-1.702 * g.astype(np.float32)))
    flat = lambda *parts: np.concatenate([np.asarray(p, dtype=np.float16).ravel() for p in parts])  # noqa
    bits = lambda v: np.resize(np.asarray(v, dtype=np.float16), N_OPS).view(np.uint16)              # noqa
    return {
        "add": (bits(flat(run_sum[:, :-1], x[:, 1:])), bits(flat(e[:, 1:], x[:, :-1]))),
        "sub": (bits(flat(s, x)), bits(flat(np.broadcast_to(mx, s.shape), np.broadcast_to(mu, x.shape)))),
        "mul": (bits(flat(d, g)), bits(flat(d, h(np.full(g.shape, 1.702))))),
        "div": (bits(flat(e, d, g)), bits(flat(np.broadcast_to(sm, e.shape), np.broadcast_to(std, d.shape), h(1 + ge)))),
        "sqrt": (bits(flat(h(var + h(1e-5)), h(np.abs(rng.normal(1, 0.3, 64))))),),
        "max": (bits(flat(s[:, 1:])), bits(flat(run_max[:, :-1]))),
        "to_int8": (bits(flat(h(h(e / sm) * 127), h(d / std))),),
        "exp": bits(flat(h(s - mx), h(-1.702 * g))),
    }


# ---------------------------------------------------------------- testbenches
def unit_tb(top, ports, n):
    """A testbench feeding `ports` ({port: (hex file, width)}) from ROMs, one entry per cycle."""
    lines = ["module tb(input clk, output [15:0] y);"]
    for p, (hexf, w) in ports.items():
        lines.append(f'  reg [15:0] r_{p} [0:{n - 1}]; initial $readmemh("{hexf}", r_{p});')
    lines.append(f"  reg [15:0] i = 0; always @(posedge clk) i <= (i == {n - 1}) ? 0 : i + 1;")
    conns = ", ".join(f".{p}(r_{p}[i][{w - 1}:0])" for p, (_, w) in ports.items())
    clk = ".clk(clk), " if top not in ("sfu_fp16_max", "sfu_fp16_to_int8") else ""
    lines.append(f"  {top} dut({clk}{conns}, .y(y));")
    lines.append("endmodule")
    return "\n".join(lines) + "\n"


def lane_exp_tb(n):
    zero16 = "16'd0"
    idle = ", ".join(f".{p}({zero16})" for p in ("max_a", "max_b", "sub_a", "sub_b", "add_a", "add_b", "div_a",
                                                  "div_b", "mul_a", "mul_b", "sqrt_a", "int_a", "ldexp_a"))
    return (f'module tb(input clk, output [15:0] y);\n'
            f'  reg [15:0] xs [0:{n - 1}]; initial $readmemh("exp_x.hex", xs);\n'
            f'  reg [15:0] t = 0; always @(posedge clk) t <= t + 1;\n'
            f'  wire [15:0] j = ((t - 1) / 7) % {n};\n'
            f"  sfu_lane dut(.clk(clk), .rst(t == 0), .exp_mode(1'b1), .exp_x(xs[j]), {idle}, .i2f_n(8'd0),\n"
            f"    .ldexp_n(8'd0),\n"
            f"    .max_y(), .sub_y(), .add_y(), .div_y(), .mul_y(), .sqrt_y(), .i2f_y(), .ldexp_y(y), .int_y());\n"
            f"endmodule\n")


def check_mapped_exp(lane, xs, workdir):
    """The mapped lane netlist (the one simulated for power) computes exp correctly:
    rtl/sfu_lane_exp_tb.v around it, against fp16_ref.exp. Raises otherwise."""
    import fp16_ref
    with open(os.path.join(workdir, "exp_x.hex"), "w") as f:
        f.write("\n".join(f"{int(x):04x}" for x in xs))
    tb = os.path.join(HERE, "rtl", "sfu_lane_exp_tb.v")
    script = (f"read_liberty -ignore_miss_func {LIB}; read_verilog {lane}; read_verilog -sv {tb}; "
              f"chparam -set N {len(xs)} sfu_lane_exp_tb; hierarchy -top sfu_lane_exp_tb; proc; memory; "
              f"flatten; sim -clock clk -zinit -n {7 * len(xs) + 76} -vcd check.vcd")
    subprocess.run(["yosys", "-q", "-p", script], cwd=workdir, check=True)
    vcd = open(os.path.join(workdir, "check.vcd")).read()
    ident = re.search(r"\$var \w+ \d+ (\S+) y_flat ", vcd).group(1)
    flat = int(re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1], 2)
    bad = [k for k, x in enumerate(xs) if (flat >> (16 * k)) & 0xFFFF != fp16_ref.exp(int(x))]
    if bad:
        raise RuntimeError(f"mapped sfu_lane computes exp wrongly for {len(bad)} of {len(xs)} inputs")
    print(f"  mapped sfu_lane: exp correct on {len(xs)} inputs", flush=True)


# ---------------------------------------------------------------- main
def main():
    f = factors()
    ops = operands()
    out = {"method": __doc__.split("\n\n")[1], "vdd_V": VDD, "freq_MHz": FREQ / 1e6, "factors_45_to_65": f,
           "units": {}, "exp": {}}
    unit_ports = {   # function -> (circuit, ports)
        "add": ("sfu_fp16_add", ["a", "b"]), "sub": ("sfu_fp16_add", ["a", "b"]),
        "mul": ("sfu_fp16_mul", ["a", "b"]), "div": ("sfu_fp16_div", ["a", "b"]),
        "sqrt": ("sfu_fp16_sqrt", ["a"]), "max": ("sfu_fp16_max", ["a", "b"]),
        "to_int8": ("sfu_fp16_to_int8", ["a"]),
    }
    clock_per_cycle = {}
    with tempfile.TemporaryDirectory() as tmp:
        for fn, (circuit, ports) in unit_ports.items():
            mapped = os.path.join(tmp, f"{circuit}.v")
            if not os.path.exists(mapped):
                mapped_netlist(UNITS[circuit][0], circuit, mapped)
            hexes = {f"{fn}_{p}.hex": ops[fn][k] for k, p in enumerate(ports)}
            tb = unit_tb(circuit, {p: (f"{fn}_{p}.hex", 16) for p in ports}, N_OPS)
            if fn == "sub":
                tb = tb.replace(f"{circuit} dut(.clk(clk), ", f"{circuit} dut(.clk(clk), .sub(1'b1), ")
            elif circuit == "sfu_fp16_add":
                tb = tb.replace(f"{circuit} dut(.clk(clk), ", f"{circuit} dut(.clk(clk), .sub(1'b0), ")
            vcd = simulate(mapped, tb, hexes, N_OPS, tmp)
            logic, clock = energy_from_vcd(mapped, vcd)
            cycles = N_OPS - 1
            clock_per_cycle[circuit] = clock / cycles
            out["units"][fn] = {"circuit": circuit, "logic_pJ_per_op": logic / cycles,
                                "clock_pJ_per_cycle": clock / cycles,
                                "energy_pJ_per_op": (logic + clock) / cycles}
            print(f"  {fn:8s} {circuit:18s} logic {logic / cycles:7.3f} pJ/op, clock {clock / cycles:6.3f} pJ/cycle",
                  flush=True)

        lane = os.path.join(tmp, "sfu_lane.v")
        mapped_netlist(LANE, "sfu_lane", lane)
        check_mapped_exp(lane, ops["exp"][:64], tmp)
        cycles = 7 * N_EXP + 1
        vcd = simulate(lane, lane_exp_tb(N_EXP), {"exp_x.hex": ops["exp"][:N_EXP]}, cycles, tmp)
        logic, clock = energy_from_vcd(lane, vcd)
        # clock of the units exp does not use (their flops are clocked in this run)
        idle_clock = (clock_per_cycle["sfu_fp16_div"] + clock_per_cycle["sfu_fp16_sqrt"]) * (cycles - 1)
        exp_energy = (logic + clock - idle_clock) / (N_EXP - 1)
        out["exp"] = {"lane_logic_pJ": logic, "lane_clock_pJ": clock, "idle_units_clock_pJ": idle_clock,
                      "energy_pJ_per_op": exp_energy, "cycles_per_op": 7}
        print(f"  exp      sfu_lane (exp mode)  {exp_energy:7.3f} pJ/op (7 cycles of the shared units)", flush=True)

    for fn, d in list(out["units"].items()) + [("exp", out["exp"])]:
        d["energy_pJ_per_op_65nm"] = d["energy_pJ_per_op"] * f["energy"]
    rows = json.load(open(os.path.join(HERE, "results", "sfu.json")))
    lane_row = next(r for r in rows if r["unit"] == "sfu_lane")
    out["leakage_mW_65nm"] = lane_row["sfu_65nm"]["leakage_mW"]
    path = os.path.join(HERE, "results", "sfu_power.json")
    with open(path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
