"""Functional check of the dequantise RTL against a Python model.

Each module is evaluated by Yosys (eval) on random and edge-case inputs,
at both characterized output widths (XW=32/Q12.20 and XW=16/Q8.8).
Exit code 0 only if every variant matches on every vector, at every
width. The internal container is a fixed 42 bits at either XW, so sh is
tested over its whole range 0..31.
"""
import os, random, re, subprocess, sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")


def clamp(r, xw):
    maxv, minv = (1 << (xw - 1)) - 1, -(1 << (xw - 1))
    return max(minv, min(maxv, r))


def ref(name, q, m, sh, zp, mw, xw):
    qz = q - zp
    half = (1 << (sh - 1)) if sh > 0 else 0
    if name == "dequant_mant_round":
        return clamp((qz * m + half) >> sh, xw)
    if name == "dequant_mant_trunc":
        return clamp((qz * m) >> sh, xw)
    if name == "dequant_pow2_round":
        return clamp((qz + half) >> sh, xw)
    if name == "dequant_pow2_trunc":
        return clamp(qz >> sh, xw)
    if name == "dequant_mant_round_zp":
        return clamp((qz * m + half) >> sh, xw)
    if name == "dequant_pow2_round_zp":
        return clamp((qz + half) >> sh, xw)
    raise KeyError(name)


def uses_mant(name):
    return "pow2" not in name


def uses_zp(name):
    return name.endswith("_zp")


def vectors(name, mw, n, rng):
    v = []
    for _ in range(n):
        q = rng.randint(-128, 127)
        m = rng.randint(0, (1 << mw) - 1) if uses_mant(name) else 1
        sh = rng.randint(0, 31)
        zp = rng.randint(-128, 127) if uses_zp(name) else 0
        v.append((q, m, sh, zp))
    # edge cases
    for q in (-128, 127, 0, 1, -1):
        for sh in (0, 1, 20, 31):
            m = (1 << mw) - 1 if uses_mant(name) else 1
            v.append((q, m, sh, 127 if uses_zp(name) else 0))
    return v


def run(name, mw, xw, vecs):
    mod = name
    params = f"-set XW {xw}" + (f" -set MW {mw}" if uses_mant(name) else "")
    lines = [f"read_verilog -sv {os.path.join(RTL, name + '.v')}", f"chparam {params} {mod}",
             f"hierarchy -top {mod}", "proc"]
    for q, m, sh, zp in vecs:
        sets = [f"-set q 8'd{q & 0xFF}", f"-set sh 6'd{sh}"]
        if uses_mant(name):
            sets.append(f"-set m {mw}'d{m}")
        if uses_zp(name):
            sets.append(f"-set zp 8'd{zp & 0xFF}")
        lines.append("eval " + " ".join(sets) + " -show x")
    out = subprocess.run(["yosys", "-p", "; ".join(l for l in lines if l)], capture_output=True, text=True).stdout
    got = []
    # Yosys eval's -show prints a plain decimal when the MSB is 0 (e.g. "520065.")
    # and a full-width binary string when the MSB is 1 (e.g. "32'111...001000.")
    # -- never a signed decimal. Both forms are handled here.
    for raw in re.findall(r"Eval result: \\x = ([^\n]+)\.", out):
        m2 = re.match(rf"{xw}'([01]+)$", raw)
        if m2:
            v = int(m2.group(1), 2)
            got.append(v - (1 << xw) if v >= (1 << (xw - 1)) else v)
        else:
            got.append(int(raw))
    return got


def main():
    rng = random.Random(7)
    ok_all = True
    cases = [("dequant_mant_round", 12), ("dequant_mant_round", 8), ("dequant_mant_round", 16),
              ("dequant_mant_trunc", 12), ("dequant_pow2_round", 0), ("dequant_pow2_trunc", 0),
              ("dequant_mant_round_zp", 12), ("dequant_pow2_round_zp", 0)]
    for xw in (32, 16):
        for name, mw in cases:
            vecs = vectors(name, mw or 1, 150, rng)
            got = run(name, mw, xw, vecs)
            exp = [ref(name, q, m, sh, zp, mw, xw) for q, m, sh, zp in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"XW={xw:2d} {name:24s} MW={mw:2d}  evaluated {len(got)}/{len(vecs)}  "
                  f"mismatches {len(bad)}  {status}")
            for v, g, e in bad[:3]:
                print("   q,m,sh,zp =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
