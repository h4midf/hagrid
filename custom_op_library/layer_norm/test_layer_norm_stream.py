"""Functional check of rtl/layer_norm_stream.v against a Python model of the same
fixed-point arithmetic (bit for bit), not against true layer-norm.

For every characterized width (Q12.20, Q8.8), lane count N (4, 8) and row length
(64, 768 and 1024), yosys simulates rtl/layer_norm_stream_tb.v on random rows (one with
outliers) and edge rows (all zero, all tied, a ramp); every output element is compared
with the model. Exit code 0 only if every element of every row matches.

Usage (from the repo root):
  python3 custom_op_library/layer_norm/test_layer_norm_stream.py
"""
import os
import random
import re
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
FILES = [os.path.join(RTL, f) for f in ("reduce_tree.v", "rsqrt_q12_20.v", "rsqrt_q8_8.v", "row_div.v",
                                        "ln_stream_lane.v", "layer_norm_stream.v", "layer_norm_stream_tb.v")]
sys.path.insert(0, HERE)
from test_layer_norm import WIDTH_CFG, clampW, rsqrt_fixed, to_signed  # noqa: E402

RS = 40
LANES = (4, 8)
ROWS = (64, 768, 1024)
EPS = 1


def ref(x, gamma, beta, eps, w):
    """layer_norm_stream.v's arithmetic over one whole row."""
    frac = WIDTH_CFG[w]["frac"]
    half = 1 << (frac - 1)
    mask = (1 << w) - 1
    recip = ((1 << RS) + len(x) // 2) // len(x)
    mean = to_signed(((sum(x) * recip) >> RS) & mask, w)
    d = [clampW(v - mean, w) for v in x]
    sq = [(v * v + half) >> frac for v in d]          # 2W bits, not clamped
    var = clampW((sum(sq) * recip) >> RS, w)
    var_eps = clampW(var + eps, w)
    inv_std = rsqrt_fixed(var_eps, w)
    norm = [clampW((v * inv_std + half) >> frac, w) for v in d]
    scaled = [clampW((v * g + half) >> frac, w) for v, g in zip(norm, gamma)]
    return [clampW(s + b, w) for s, b in zip(scaled, beta)]


def write_hex(path, values, n, w):
    with open(path, "w") as f:
        for k in range(0, len(values), n):
            word = 0
            for i, v in enumerate(values[k:k + n]):
                word |= (v & ((1 << w) - 1)) << (i * w)
            f.write(f"{word:0{n * w // 4}x}\n")


def top_signals(vcd):
    """{name: id} of the testbench's own signals (the first scope)."""
    ids, depth = {}, 0
    for line in vcd.splitlines():
        if line.startswith("$scope"):
            depth += 1
        elif line.startswith("$upscope"):
            depth -= 1
        elif line.startswith("$var") and depth == 1:
            parts = line.split()
            ids[parts[4]] = parts[3]
        elif line.startswith("$enddefinitions"):
            break
    return ids


def outputs(vcd, n, w, step_name):
    """The output row: y lanes at every time y_valid is high, keyed by the signal
    `step_name` (the element group at the output), the last value per group."""
    ids = top_signals(vcd)
    names = {ids["y_flat"]: "y_flat", ids["y_valid"]: "y_valid", ids[step_name]: "step"}
    state, got = {}, {}
    body = vcd.split("$enddefinitions", 1)[1]
    for line in body.splitlines():
        if line.startswith("#"):
            if state.get("y_valid") == 1 and "y_flat" in state:
                got[state["step"]] = state["y_flat"]
            continue
        m = re.match(r"^b([01xz]+) (\S+)$", line)
        if m and m.group(2) in names:
            if "x" in m.group(1) or "z" in m.group(1):
                state[names[m.group(2)]] = None
            else:
                state[names[m.group(2)]] = int(m.group(1), 2)
            continue
        m = re.match(r"^([01xz])(\S+)$", line)
        if m and m.group(2) in names:
            state[names[m.group(2)]] = int(m.group(1)) if m.group(1) in "01" else None
    if state.get("y_valid") == 1 and "y_flat" in state:
        got[state["step"]] = state["y_flat"]
    out = []
    for step in sorted(got):
        word = got[step]
        if word is None:
            raise RuntimeError(f"y_flat undefined at step {step}")
        out += [to_signed((word >> (i * w)) & ((1 << w) - 1), w) for i in range(n)]
    return out


def simulate(x, gamma, beta, n, w, row):
    frac = WIDTH_CFG[w]["frac"]
    with tempfile.TemporaryDirectory() as tmp:
        write_hex(os.path.join(tmp, "x.hex"), x, n, w)
        write_hex(os.path.join(tmp, "gamma.hex"), gamma, n, w)
        write_hex(os.path.join(tmp, "beta.hex"), beta, n, w)
        vcd_path = os.path.join(tmp, "out.vcd")
        cycles = 3 * (row // n) + 6
        script = (f"read_verilog -sv {' '.join(FILES)}; "
                  f"chparam -set N {n} -set W {w} -set FRAC {frac} -set ROW {row} -set EPS {EPS} "
                  f"layer_norm_stream_tb; hierarchy -top layer_norm_stream_tb; proc; flatten; opt_clean; "
                  f"sim -clock clk -zinit -n {cycles} -vcd {vcd_path}")
        subprocess.run(["yosys", "-q", "-p", script], check=True, cwd=tmp)
        with open(vcd_path) as f:
            return outputs(f.read(), n, w, "step")


def rows_for(row, w, rng):
    scale = 1 << WIDTH_CFG[w]["frac"]
    big = (1 << (w - 1)) - 1
    gamma = [rng.randint(scale // 4, 2 * scale) for _ in range(row)]
    beta = [rng.randint(-scale, scale) for _ in range(row)]
    rand = [rng.randint(-10 * scale, 10 * scale) for _ in range(row)]
    outl = list(rand)
    for k in rng.sample(range(row), 4):
        outl[k] = rng.choice((-1, 1)) * rng.randint(big // 4, big)
    return [
        ("random", rand, gamma, beta),
        ("outliers", outl, gamma, beta),
        ("zero", [0] * row, [scale] * row, [0] * row),
        ("tied", [scale] * row, [scale] * row, [0] * row),
        ("ramp", list(range(1, row + 1)), [scale] * row, [0] * row),
    ]


def main():
    rng = random.Random(0)
    bad, total = [], 0
    for w in (32, 16):
        for n in LANES:
            for row in ROWS:
                for name, x, gamma, beta in rows_for(row, w, rng):
                    total += 1
                    got = simulate(x, gamma, beta, n, w, row)
                    exp = ref(x, gamma, beta, EPS, w)
                    ok = got == exp
                    print(f"  W={w} N={n} ROW={row} {name}: {'ok' if ok else 'MISMATCH'}", flush=True)
                    if not ok:
                        diff = [k for k in range(min(len(got), len(exp))) if got[k] != exp[k]]
                        bad.append((w, n, row, name, len(got), len(exp), diff[:5]))
    print(f"layer_norm_stream: {total - len(bad)}/{total} rows match")
    for b in bad:
        print(f"  {b}")
    sys.exit(0 if not bad else 1)


if __name__ == "__main__":
    main()
