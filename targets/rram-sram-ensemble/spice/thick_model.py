"""3.3 V device models (nhv, phv): PTM 65 nm bulk with the oxide, threshold and gate
tunneling of a thick-oxide I/O device (design.json "model_assumptions.thick_oxide"), since
PTM has no I/O transistor. Also the core devices' model path and the access transistor's
on-resistance."""
import json
import os
import re
import subprocess
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "..")
MODELS = os.path.abspath(os.path.join(TARGET, "..", "hyflexpim", "spice", "models", "65nm_bulk.pm"))
DESIGN = json.load(open(os.path.join(TARGET, "default", "design.json")))
THICK = DESIGN["model_assumptions"]["thick_oxide"]
DTOX_NM = 0.65   # PTM's toxe - toxp


def _set(text: str, name: str, value: str) -> str:
    new, count = re.subn(rf"(\b{name}\s*=\s*)[-+\d.eE]+", rf"\g<1>{value}", text, flags=re.I)
    if count == 0:
        raise ValueError(f"{name} not found in {MODELS}")
    return new


def thick_model_text() -> str:
    text = open(MODELS).read()
    split = re.search(r"\.model\s+pmos\s+pmos", text).start()
    parts = []
    for part, name, kind, sign in ((text[:split], "nhv", "nmos", 1), (text[split:], "phv", "pmos", -1)):
        part = re.sub(rf"\.model\s+{kind}\s+{kind}", f".model {name} {kind}", part)
        for key in ("toxe", "toxm", "toxref"):
            part = _set(part, key, f"{THICK['tox_nm']}e-9")
        part = _set(part, "toxp", f"{THICK['tox_nm'] - DTOX_NM}e-9")
        part = _set(part, "vth0", f"{sign * THICK['vth_V']}")
        part = _set(part, "igcmod", "0")
        part = _set(part, "igbmod", "0")
        parts.append(part)
    return "".join(parts)


def run(netlist: str) -> str:
    """ngspice output of a netlist; __THICK__ in it names the thick-oxide model file."""
    with tempfile.TemporaryDirectory() as tmp:
        model = os.path.join(tmp, "thick.pm")
        with open(model, "w") as f:
            f.write(thick_model_text())
        path = os.path.join(tmp, "net.sp")
        with open(path, "w") as f:
            f.write(netlist.replace("__THICK__", model))
        return subprocess.run(["ngspice", "-b", path], capture_output=True, text=True).stdout


def get(out: str, key: str) -> float:
    m = re.search(rf"^{key}\s*=\s*([-+\d.eE]+)", out, re.M)
    if not m:
        raise RuntimeError(f"ngspice: no '{key}'\n{out[-2000:]}")
    return float(m.group(1))


def access_on_ohm() -> float:
    """On-resistance of the access transistor, its gate at the WL's high voltage."""
    wl = DESIGN["model_assumptions"]["wl"]
    out = run(f""".include __THICK__
VG g 0 {wl['high_V']}
VD d 0 0.1
MA d g 0 0 nhv w={wl['access_w_nm']}n l={THICK['l_nm']}n
.control
op
print i(vd)
.endc
.end
""")
    return 0.1 / -get(out, "i\\(vd\\)")
