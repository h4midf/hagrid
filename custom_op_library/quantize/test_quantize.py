"""Functional check of the quantise RTL against a Python model.

Each module is evaluated by Yosys (eval) on random and edge-case inputs, at
XW=32 (Q12.20) and XW=16 (Q8.8). Exit code 0 only if every vector matches.
"""
import os, random, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")


def clamp8(r):
    return max(-128, min(127, r))


def ref(name, x, m, sh, zp, mw):
    half = (1 << (sh - 1)) if sh > 0 else 0
    if name == "quant_mant_round":
        return clamp8((x * m + half) >> sh)
    if name == "quant_mant_trunc":
        return clamp8((x * m) >> sh)
    if name == "quant_pow2_round":
        return clamp8((x + half) >> sh)
    if name == "quant_pow2_trunc":
        return clamp8(x >> sh)
    if name == "quant_mant_round_zp":
        return clamp8(((x * m + half) >> sh) + zp)
    if name == "quant_pow2_round_zp":
        return clamp8(((x + half) >> sh) + zp)
    raise KeyError(name)


def uses_mant(name):
    return "pow2" not in name


def uses_zp(name):
    return name.endswith("_zp")


def max_valid_sh(name, mw, xw):
    """Largest sh: the rounding term 2^(sh-1) must fit in the module's intermediate
    container (1+XW+MW bits for the mantissa variants, XW for pow2), rtl/README.md."""
    container = (1 + xw + mw) if uses_mant(name) else xw
    return min(31, container - 2)


def vectors(name, mw, xw, n, rng):
    sh_max = max_valid_sh(name, mw, xw)
    v = []
    for _ in range(n):
        x = (rng.randint(-(1 << (xw - 1)), (1 << (xw - 1)) - 1) if rng.random() < 0.5
             else rng.randint(-(1 << (xw - 1)) // 8, (1 << (xw - 1)) // 8))
        m = rng.randint(0, (1 << mw) - 1) if uses_mant(name) else 1
        sh = rng.randint(0, sh_max)
        zp = rng.randint(-128, 127) if uses_zp(name) else 0
        v.append((x, m, sh, zp))
    # edge cases
    maxv, minv = (1 << (xw - 1)) - 1, -(1 << (xw - 1))
    for x in (minv, maxv, 0, 1, -1):
        for sh in (0, 1, sh_max // 2, sh_max):
            m = (1 << mw) - 1 if uses_mant(name) else 1
            v.append((x, m, sh, 127 if uses_zp(name) else 0))
    return v


def run(name, mw, xw, vecs):
    mod = name
    params = f"-set XW {xw}" + (f" -set MW {mw}" if uses_mant(name) else "")
    lines = [f"read_verilog -sv {os.path.join(RTL, name + '.v')}", f"chparam {params} {mod}",
             f"hierarchy -top {mod}", "proc"]
    for x, m, sh, zp in vecs:
        sets = [f"-set x {xw}'d{x & ((1 << xw) - 1)}", f"-set sh 6'd{sh}"]
        if uses_mant(name):
            sets.append(f"-set m {mw}'d{m}")
        if uses_zp(name):
            sets.append(f"-set zp 8'd{zp & 0xFF}")
        lines.append("eval " + " ".join(sets) + " -show q")
    out = subprocess.run(["yosys", "-p", "; ".join(l for l in lines if l)], capture_output=True, text=True).stdout
    got = []
    for b in re.findall(r"Eval result: \\q = 8'([01]+)", out):
        v = int(b, 2)
        got.append(v - 256 if v >= 128 else v)
    return got


def main():
    rng = random.Random(7)
    ok_all = True
    cases = [("quant_mant_round", 12), ("quant_mant_round", 8), ("quant_mant_round", 16),
             ("quant_mant_trunc", 12), ("quant_pow2_round", 0), ("quant_pow2_trunc", 0),
             ("quant_mant_round_zp", 12), ("quant_pow2_round_zp", 0)]
    for xw in (32, 16):
        for name, mw in cases:
            vecs = vectors(name, mw or 1, xw, 150, rng)
            got = run(name, mw, xw, vecs)
            exp = [ref(name, x, m, sh, zp, mw) for x, m, sh, zp in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"XW={xw:2d} {name:22s} MW={mw:2d}  evaluated {len(got)}/{len(vecs)}  "
                  f"mismatches {len(bad)}  {status}")
            for v, g, e in bad[:3]:
                print("   x,m,sh,zp =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
