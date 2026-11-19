"""Functional check of quark_gelu.v against a Python model of the same fixed-point
arithmetic (QUARK's Eq.11-14, the sigmoid through test_quark_softmax_row8.py's model).
Yosys eval on random and edge-case inputs around |x| = 2.4, at both widths. Exit code
0 only if every vector matches."""
import importlib.util
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = [os.path.join(HERE, "rtl", f) for f in
       ("quark_appro_exp.v", "quark_appro_ln.v", "reduce_tree.v", "quark_softmax_row8.v", "quark_gelu.v")]

spec = importlib.util.spec_from_file_location("test_softmax", os.path.join(HERE, "test_quark_softmax_row8.py"))
sm = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sm)

CONSTS = {32: dict(frac=20, thresh=2516582, c1702=1784739, sentinel=-1048576000),
          16: dict(frac=8, thresh=614, c1702=436, sentinel=-8000)}


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r, w):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    return max(minv, min(maxv, r))


def ref(x, w):
    c = CONSTS[w]
    frac, thresh, c1702, sentinel = c["frac"], c["thresh"], c["c1702"], c["sentinel"]
    half = 1 << (frac - 1)

    m = (x * c1702 + half) >> frac
    s1 = clampW(-m, w)
    row = [0, s1, sentinel, sentinel, sentinel, sentinel, sentinel, sentinel]
    p = sm.ref_row(row, w, frac)
    p0 = p[0]

    inner = clampW((x * p0 + half) >> frac, w)
    outer = max(x, 0)
    abs_x = -x if x < 0 else x
    return inner if abs_x < thresh else clampW(outer, w)


def vectors(w, frac, n, rng):
    maxv = (1 << (w - 1)) - 1
    scale = 1 << frac
    thresh = CONSTS[w]["thresh"]
    v = [rng.randint(-maxv // 4, maxv // 4) for _ in range(n)]
    for k in (thresh - 2, thresh - 1, thresh, thresh + 1, thresh + 2, -thresh - 1, -thresh, -thresh + 1):
        v.append(k)
    v += [0, scale, -scale]
    return v


def run(w, frac, vecs):
    lines = [f"read_verilog -sv {' '.join(RTL)}", f"chparam -set W {w} -set FRAC {frac} quark_gelu",
             "hierarchy -top quark_gelu", "proc -norom", "flatten", "select quark_gelu"]
    for x in vecs:
        lines.append(f"eval -set x {w}'d{x & ((1 << w) - 1)} -show y")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\y = ([^\n]+)\.", out):
        m = re.match(rf"{w}'([01]+)$", raw)
        got.append(to_signed(int(m.group(1), 2), w) if m else int(raw))
    return got


def main():
    rng = random.Random(131)
    ok_all = True
    for w, frac in ((32, 20), (16, 8)):
        vecs = vectors(w, frac, 60, rng)
        got = run(w, frac, vecs)
        exp = [ref(x, w) for x in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"quark_gelu W={w:2d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:5]:
            print("   x =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
