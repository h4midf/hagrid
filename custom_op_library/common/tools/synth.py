"""Shared synthesis: yosys (synth, dfflibmap, ABC mapping) with an ABC timing report.

synth(files, top, params) returns area (um^2), critical-path delay (ps) and cell counts.
Only the Verilog files passed in are read; nothing is generated.
"""
import os
import re
import signal
import subprocess
import tempfile

from liberty import CELLS, LIB

_HERE = os.path.dirname(os.path.abspath(__file__))
_LATCH_MAP = os.path.join(_HERE, "cells_latch.v")   # latches -> DLH_X1 / DLL_X1
_ABC_SCRIPT = os.path.join(_HERE, "abc_timing.script")
with open(_ABC_SCRIPT, "w") as _f:
    _f.write("strash\ndch\nmap\ntopo\nstime -p\n")
# sized=True: the same mapping, then ABC's buffering and gate sizing before timing.
# Without them a high-fanout net is driven by one X1 gate (e.g. a 44-load mux select
# costs ~380 ps), which no real flow would leave; used where a clock target has to
# be met (targets/hyflexpim/sfu).
_ABC_SIZED_SCRIPT = os.path.join(_HERE, "abc_timing_sized.script")
with open(_ABC_SIZED_SCRIPT, "w") as _f:
    _f.write("strash\ndch\nmap\ntopo\nbuffer -p\nupsize\ndnsize\nstime -p\n")

# ABC's `dch` can run for an unbounded time on some circuits: a run that exceeds this
# is killed and raises RuntimeError, like any other synthesis failure
DEFAULT_TIMEOUT_S = 1200


def synth(files, top, params=None, timeout=DEFAULT_TIMEOUT_S, sized=False):
    params = params or {}
    abc_script = _ABC_SIZED_SCRIPT if sized else _ABC_SCRIPT
    chp = " ".join(f"-set {k} {v}" for k, v in params.items())
    reads = " ".join(files)
    stat_fd, stat_path = tempfile.mkstemp(suffix=".txt")
    os.close(stat_fd)
    script = (f"read_verilog -sv {reads}; "
              + (f"chparam {chp} {top}; " if chp else "")
              + f"hierarchy -top {top}; synth -flatten -top {top}; "
              f"dfflibmap -liberty {LIB}; techmap -map {_LATCH_MAP}; "
              f"abc -liberty {LIB} -script {abc_script}; opt_clean; "
              f"tee -o {stat_path} stat -liberty {LIB}")
    log, stat = _run(script, top, params, timeout, stat_path)
    m = re.search(r"Chip area for module.*?:\s*([\d.]+)", stat)
    if not m:
        raise RuntimeError(f"synthesis failed for {top} {params}:\n{log[-2000:]}")
    delays = re.findall(r"Delay =\s+([\d.]+) ps", log)
    cells = {}
    # the per-type area is printed in scientific notation from 1000 um^2 up
    # ("472 2.13E+03 DFF_X1"); without the exponent those rows were dropped
    for c in re.finditer(r"^\s+(\d+)\s+[\d.]+(?:[eE][+-]?\d+)?\s+(\w+)\s*$", stat, flags=re.M):
        if c.group(2) in CELLS:
            cells[c.group(2)] = cells.get(c.group(2), 0) + int(c.group(1))
    return {"area_um2": float(m.group(1)), "delay_ps": float(delays[-1]) if delays else None, "cells": cells}


def _module_base(name):
    """`$paramod$<hash>\\lane` / `\\lane` -> `lane`."""
    return name.rsplit("\\", 1)[-1]


def synth_hier(files, top, params, timeout=DEFAULT_TIMEOUT_S):
    """Hierarchical synthesis: the same flow as synth() but without flattening, so ABC
    maps every module on its own (a module instantiated N times is mapped once). For
    circuits whose flat netlist ABC cannot map in time.

    Returns area_um2 and cells of the whole design (every instance counted), and
    module_delays_ps: each module's own ABC critical path ({module name: ps}, the
    parameterized name reduced to the module's name; the top module's own logic under
    `top`). No path is timed across module boundaries: a caller composes a clock period
    from the module delays along its register-to-register paths."""
    chp = " ".join(f"-set {k} {v}" for k, v in params.items())
    stat_fd, stat_path = tempfile.mkstemp(suffix=".txt")
    os.close(stat_fd)
    script = (f"read_verilog -sv {' '.join(files)}; "
              + (f"chparam {chp} {top}; " if chp else "")
              + f"synth -top {top}; rename -top {top}; "
              f"dfflibmap -liberty {LIB}; techmap -map {_LATCH_MAP}; "
              f"abc -liberty {LIB} -script {_ABC_SCRIPT}; opt_clean; "
              f"tee -o {stat_path} stat -liberty {LIB} -top {top}")
    log, stat = _run(script, top, params, timeout, stat_path)
    # the last ABC run per module is the mapping above (synth's own ABC prints no Delay)
    delays, current = {}, None
    for line in log.splitlines():
        m = re.search(r"Extracting gate netlist of module `(.+?)'", line)
        if m:
            current = _module_base(m.group(1))
            continue
        m = re.search(r"Delay =\s+([\d.]+) ps", line)
        if m and current is not None:
            delays[current] = float(m.group(1))
    hier = stat.split("=== design hierarchy ===", 1)
    if len(hier) != 2:
        raise RuntimeError(f"hierarchical synthesis failed for {top} {params}:\n{log[-2000:]}")
    # the design hierarchy section counts every cell of every instance
    cells = {}
    for c in re.finditer(r"^\s+(\d+)\s+[\d.]+(?:[eE][+-]?\d+)?\s+(\w+)\s*$", hier[1], flags=re.M):
        if c.group(2) in CELLS:
            cells[c.group(2)] = cells.get(c.group(2), 0) + int(c.group(1))
    if not cells or top not in delays:
        raise RuntimeError(f"hierarchical synthesis of {top} {params}: no cells or no top-level delay")
    # a module of the design that ABC never mapped has no logic gates (e.g. a constant
    # multiply by a power of two, reduced to wiring): no combinational delay
    for name in re.findall(r"^=== (.+?) ===$", hier[0], flags=re.M):
        delays.setdefault(_module_base(name), 0.0)
    area = sum(CELLS[c]["area"] * n for c, n in cells.items())
    return {"area_um2": area, "cells": cells, "module_delays_ps": delays}


def _run(script, top, params, timeout, stat_path):
    """Runs yosys on `script`; returns (log, the stat file's text). Kills the whole
    process group and raises if it exceeds `timeout` seconds."""
    # its own process group, so a timeout kills yosys-abc too (killing yosys alone
    # leaves its yosys-abc child running)
    proc = subprocess.Popen(["yosys", "-p", script], stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                             text=True, start_new_session=True)
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        proc.communicate()
        try:
            os.unlink(stat_path)
        except OSError:
            pass
        raise RuntimeError(f"synthesis for {top} {params} did not complete within {timeout}s -- killed")
    log = stdout + stderr
    with open(stat_path) as f:
        stat = f.read()
    os.unlink(stat_path)
    return log, stat
