"""Functional check of rtl/rram_decoder.v against a Python model of its FSM: the state,
ADC enable, done and column select of every cycle, and the write selects.

Usage (from the repo root):
  python3 targets/rram-sram-ensemble/periphery/test_rram_decoder.py
"""
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", "rram_decoder.v"), os.path.join(HERE, "rtl", "rram_decoder_tb.v")]
CYCLES, ROW, COL = 24, 37, 5
S_INPUT, S_COMPUTE, S_WRITE = 0, 1, 2


def ref() -> tuple:
    state = cnt = 0
    trace = 0
    for t in range(CYCLES):
        adc = state == S_COMPUTE
        word = (state << 10) | (int(adc) << 9) | (int(adc and cnt == 7) << 8) | ((1 << cnt) if adc else 0)
        trace = (trace << 12) | word
        rst, go, write_req = t < 2, t == 3, 15 <= t < 18
        if rst:
            state, cnt = S_INPUT, 0
        elif state == S_INPUT:
            if write_req:
                state = S_WRITE
            elif go:
                state, cnt = S_COMPUTE, 0
        elif state == S_COMPUTE:
            if cnt == 7:
                state = S_INPUT
            cnt = (cnt + 1) % 8
        elif not write_req:
            state = S_INPUT
    return trace, ((1 << ROW) << 128) | ((1 << COL) << 64) | (1 << COL)


def last_value(vcd: str, name: str) -> int:
    ident = re.search(rf"\$var \w+ \d+ (\S+) {name} ", vcd).group(1)
    last = re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1]
    if "x" in last:
        raise RuntimeError(f"{name} is undefined at the end of the simulation")
    return int(last, 2)


def main() -> None:
    fd, vcd_path = tempfile.mkstemp(suffix=".vcd")
    os.close(fd)
    script = (f"read_verilog -sv {' '.join(RTL)}; hierarchy -top rram_decoder_tb; proc; flatten; opt_clean; "
              f"sim -clock clk -zinit -n {CYCLES + 2} -vcd {vcd_path}")
    subprocess.run(["yosys", "-q", "-p", script], check=True)
    vcd = open(vcd_path).read()
    os.unlink(vcd_path)
    want_trace, want_sel = ref()
    got_trace, got_sel = last_value(vcd, "trace"), last_value(vcd, "sel_cap")
    if got_trace != want_trace:
        raise RuntimeError(f"trace mismatch\n got  {got_trace:0{CYCLES * 3}x}\n want {want_trace:0{CYCLES * 3}x}")
    if got_sel != want_sel:
        raise RuntimeError(f"write selects mismatch: got {got_sel:x}, want {want_sel:x}")
    print(f"rram_decoder: {CYCLES} cycles and the write selects match")


if __name__ == "__main__":
    main()
