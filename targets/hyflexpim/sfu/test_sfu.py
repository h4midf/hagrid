"""Bit-exact check of every SFU unit (rtl/sfu_*.v, evaluated combinationally in yosys)
against fp16_ref.py: random operands, operands with close exponents, and edge cases
(zeros, subnormals, the largest and smallest normals, overflow, underflow).
Exit code 0 only if every vector of every unit matches.

Usage: python3 targets/hyflexpim/sfu/test_sfu.py [unit ...]   (default: all units)
"""
import random
import sys

import fp16_ref
from yosys_eval import evaluate

EDGES = [0x0000, 0x8000, 0x0001, 0x83FF, 0x0400, 0x8400, 0x3C00, 0xBC00, 0x7BFF, 0xFBFF,
         0x7C00, 0x3C01, 0xBBFF, 0x0401, 0x8401, 0x7800, 0x0800, 0x4000, 0x3800, 0x5BFF]


def pairs(n, seed):
    """Edge-case pairs, then n random pairs, half of them with exponents within 2."""
    rng = random.Random(seed)
    out = [(a, b) for a in EDGES for b in EDGES]
    for _ in range(n):
        a = rng.getrandbits(16)
        if rng.random() < 0.5:
            e = (((a >> 10) & 0x1F) + rng.randint(-2, 2)) & 0x1F
            b = (rng.getrandbits(1) << 15) | (e << 10) | rng.getrandbits(10)
        else:
            b = rng.getrandbits(16)
        out.append((a, b))
    return out


def singles(n, seed):
    rng = random.Random(seed)
    return EDGES + [rng.getrandbits(16) for _ in range(n)]


def h(v):
    return f"16'd{v}"


# unit name -> (rtl modules, top, vectors, input setter, output, reference[, pipelined])
UNITS = {
    "add": (["sfu_prefix_add", "sfu_fp16_add"], "sfu_fp16_add", pairs(1200, 1),
            lambda v: {"a": h(v[0]), "b": h(v[1]), "sub": "1'd0"}, "y", lambda v: fp16_ref.add(*v)),
    "sub": (["sfu_prefix_add", "sfu_fp16_add"], "sfu_fp16_add", pairs(1200, 2),
            lambda v: {"a": h(v[0]), "b": h(v[1]), "sub": "1'd1"}, "y", lambda v: fp16_ref.sub(*v)),
    "mul": (["sfu_prefix_add", "sfu_csa_mul", "sfu_fp16_mul"], "sfu_fp16_mul", pairs(1200, 3),
            lambda v: {"a": h(v[0]), "b": h(v[1])}, "y", lambda v: fp16_ref.mul(*v)),
    "div": (["sfu_prefix_add", "sfu_fp16_div"], "sfu_fp16_div", pairs(1200, 4),
            lambda v: {"a": h(v[0]), "b": h(v[1])}, "y", lambda v: fp16_ref.div(*v)),
    "sqrt": (["sfu_prefix_add", "sfu_fp16_sqrt"], "sfu_fp16_sqrt", singles(1500, 5),
             lambda v: {"a": h(v)}, "y", fp16_ref.sqrt),
    # every int8
    "int8_to_fp16": (["sfu_int8_to_fp16"], "sfu_int8_to_fp16", list(range(-128, 128)),
                     lambda v: {"n": f"8'd{v & 0xFF}"}, "y", fp16_ref.int8_to_fp16, False),
    # random and edge-case values, each with random and extreme shifts
    "ldexp": (["sfu_fp16_ldexp"], "sfu_fp16_ldexp",
              [(a, n) for a, n in zip(singles(1200, 9),
                                      [random.Random(10 + i).randint(-128, 127) if i % 4 else
                                       (-128, -20, 0, 20, 127)[i % 5] for i in range(1220)])],
              lambda v: {"a": h(v[0]), "n": f"8'd{v[1] & 0xFF}"}, "y", lambda v: fp16_ref.ldexp(*v), False),
    "max": (["sfu_prefix_add", "sfu_fp16_max"], "sfu_fp16_max", pairs(1200, 6),
            lambda v: {"a": h(v[0]), "b": h(v[1])}, "y", lambda v: fp16_ref.fmax(*v), False),
    # every input from -256 to 256 in steps the format can hold near the rounding edges
    "to_int8": (["sfu_fp16_to_int8"], "sfu_fp16_to_int8",
                singles(1000, 7) + [x | s for x in range(0x3400, 0x5C40) for s in (0, 0x8000)][::7],
                lambda v: {"a": h(v)}, "y", lambda v: fp16_ref.to_int8(v) & 0xFF, False),
}


def run(name):
    modules, top, vecs, setter, output, ref, *rest = UNITS[name]
    pipelined = rest[0] if rest else True
    got = evaluate(["sfu_pipe"] + modules, top, [setter(v) for v in vecs], output, pipelined)
    bad = [(v, g, ref(v)) for v, g in zip(vecs, got) if g != ref(v)]
    print(f"{name:8s} {top}: {len(vecs)} vectors, {len(bad)} mismatches")
    for v, g, e in bad[:5]:
        print(f"   in={[hex(x) for x in (v if isinstance(v, tuple) else (v,))]} rtl={g:#06x} ref={e:#06x}")
    return not bad


LANE_RTL = ["sfu_pipe", "sfu_prefix_add", "sfu_csa_mul", "sfu_fp16_add", "sfu_fp16_mul", "sfu_fp16_div",
            "sfu_fp16_sqrt", "sfu_fp16_max", "sfu_fp16_to_int8", "sfu_int8_to_fp16", "sfu_fp16_ldexp",
            "sfu_lane", "sfu_lane_exp_tb"]


def run_exp_lane():
    """exp on the lane's shared units (sfu_lane exp mode, cycle-accurate yosys simulation)
    against fp16_ref.exp: 3 x 64 inputs, random, edge cases and a sweep over |x| < 16."""
    import os
    import re
    import subprocess
    import tempfile
    from yosys_eval import RTL
    rng = random.Random(11)
    sweep = [x | s for x in range(0x0400, 0x4C00, 0x4800 // 64) for s in (0, 0x8000)]
    batches = [[rng.getrandbits(16) for _ in range(64)], (EDGES * 4)[:64], sweep[:64]]
    bad, total = [], 0
    for xs in batches:
        with tempfile.TemporaryDirectory() as tmp:
            with open(os.path.join(tmp, "exp_x.hex"), "w") as f:
                f.write("\n".join(f"{x:04x}" for x in xs))
            files = " ".join(os.path.join(RTL, m + ".v") for m in LANE_RTL)
            script = (f"read_verilog -sv {files}; hierarchy -top sfu_lane_exp_tb; proc; memory; flatten; "
                      f"opt_clean; sim -clock clk -zinit -n {7 * len(xs) + 76} -vcd y.vcd")
            subprocess.run(["yosys", "-q", "-p", script], cwd=tmp, check=True)
            vcd = open(os.path.join(tmp, "y.vcd")).read()
        ident = re.search(r"\$var \w+ \d+ (\S+) y_flat ", vcd).group(1)
        flat = int(re.findall(rf"^b([01x]+) {re.escape(ident)}$", vcd, flags=re.M)[-1], 2)
        for k, x in enumerate(xs):
            got, exp = (flat >> (16 * k)) & 0xFFFF, fp16_ref.exp(x)
            total += 1
            if got != exp:
                bad.append((x, got, exp))
    print(f"exp      sfu_lane (exp mode): {total} vectors, {len(bad)} mismatches")
    for x, g, e in bad[:5]:
        print(f"   x={x:#06x} rtl={g:#06x} ref={e:#06x}")
    return not bad


def main():
    names = sys.argv[1:] or list(UNITS) + ["exp"]
    ok = all([run_exp_lane() if n == "exp" else run(n) for n in names])
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
