"""Runs the target's CACTI 6.5 (targets/hyflexpim/CACTI) on one SRAM: writes a config
from CACTI's own cache.cfg with the given options and parses area, read energy,
leakage and timing from its output.

Build CACTI first (from targets/hyflexpim/CACTI):
  mkdir -p obj_opt && make -f cacti.mk TAG=opt CXX="clang++ -std=c++98" CC=clang \
      OPT="-O2 -DNTHREADS=8 -w"
"""
import os
import re
import subprocess
import tempfile

CACTI_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "CACTI")
BINARY = os.path.join(CACTI_DIR, "cacti")


def _set(cfg, key, value):
    cfg, n = re.subn(rf"^{re.escape(key)}.*$", f"{key} {value}", cfg, count=1, flags=re.M)
    if n != 1:
        raise KeyError(f"CACTI config has no line '{key}'")
    return cfg


def _num(pattern, out):
    m = re.search(pattern, out)
    return float(m.group(1)) if m else None


def run(size_bytes, block_bytes, node_nm, cell="itrs-hp", peripheral="itrs-hp", ecc=False):
    """One CACTI run of a direct-mapped scratch RAM with one read-write port; None if
    CACTI finds no organization for it."""
    if not os.path.exists(BINARY):
        raise FileNotFoundError(f"{BINARY}: build CACTI first (see this module's docstring)")
    with open(os.path.join(CACTI_DIR, "cache.cfg")) as f:
        cfg = f.read()
    for key, value in (("-size (bytes)", size_bytes), ("-block size (bytes)", block_bytes),
                       ("-associativity", 1), ("-read-write port", 1), ("-exclusive read port", 0),
                       ("-exclusive write port", 0), ("-technology (u)", node_nm / 1000),
                       ("-output/input bus width", block_bytes * 8), ("-cache type", '"ram"'),
                       ("-Data array cell type -", f'"{cell}"'), ("-Data array peripheral type -", f'"{peripheral}"'),
                       ("-Add ECC -", '"true"' if ecc else '"false"')):
        cfg = _set(cfg, key, value)
    fd, path = tempfile.mkstemp(suffix=".cfg", dir=CACTI_DIR)
    with os.fdopen(fd, "w") as f:
        f.write(cfg)
    try:
        out = subprocess.run([BINARY, "-infile", os.path.basename(path)], cwd=CACTI_DIR,
                             capture_output=True, text=True).stdout
    finally:
        os.unlink(path)
    area = _num(r"Data array: Area \(mm2\): ([\d.e+-]+)", out)
    if area is None:
        return None
    return {"size_bytes": size_bytes, "block_bytes": block_bytes, "node_nm": node_nm,
            "area_mm2": area,
            "read_energy_pJ": _num(r"Total dynamic read energy per access \(nJ\): ([\d.e+-]+)", out) * 1e3,
            "leakage_mW": _num(r"Total leakage power of a bank \(mW\): ([\d.e+-]+)", out),
            "access_ns": _num(r"Access time \(ns\): ([\d.e+-]+)", out),
            "cycle_ns": _num(r"Cycle time \(ns\):\s+([\d.e+-]+)", out)}


def smallest(size_bytes, node_nm, widths, **kw):
    """The run with the smallest area over the access widths CACTI accepts."""
    runs = [r for w in widths if w <= size_bytes for r in [run(size_bytes, w, node_nm, **kw)] if r]
    if not runs:
        raise RuntimeError(f"CACTI found no organization for {size_bytes} B at {node_nm} nm")
    return min(runs, key=lambda r: r["area_mm2"])
