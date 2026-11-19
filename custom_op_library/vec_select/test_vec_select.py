"""Functional check of vec_select against a Python model (out[i] = mask[i] ? x[i] :
fill, exact: no arithmetic). Each vector is evaluated by Yosys (eval) on random and
edge-case N-lane rows, at both data widths. Exit code 0 only if every vector matches,
for every N and W."""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "vec_select.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def pack(lanes, w):
    v = 0
    for i, lane in enumerate(lanes):
        v |= (lane & ((1 << w) - 1)) << (i * w)
    return v


def unpack(v, n, w):
    return tuple(to_signed((v >> (i * w)) & ((1 << w) - 1), w) for i in range(n))


def ref(x, mask, fill):
    return tuple(xi if (mask >> i) & 1 else fill for i, xi in enumerate(x))


def vectors(n, w, count, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    v = []
    for _ in range(count):
        x = tuple(rng.randint(minv, maxv) for _ in range(n))
        v.append((x, rng.randint(0, (1 << n) - 1), rng.randint(minv, maxv)))
    x = tuple(rng.randint(minv, maxv) for _ in range(n))
    v += [(x, 0, minv), (x, (1 << n) - 1, minv), (x, 0, maxv), (x, 0b0101 & ((1 << n) - 1), minv),
          ((maxv,) * n, 0, minv), ((minv,) * n, (1 << n) - 1, maxv)]
    return v


def run(n, w, vecs):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set N {n} -set W {w} vec_select",
             "hierarchy -top vec_select", "proc"]
    for x, mask, fill in vecs:
        lines.append(f"eval -set x_flat {n * w}'d{pack(x, w)} -set mask {n}'d{mask} "
                     f"-set fill {w}'d{fill & ((1 << w) - 1)} -show out_flat")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\out_flat = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        got.append(unpack(int(m.group(2), 2) if m else int(raw), n, w))
    return got


def main():
    rng = random.Random(23)
    ok_all = True
    for w in (32, 16):
        for n in (1, 8, 64):
            vecs = vectors(n, w, 40, rng)
            got = run(n, w, vecs)
            exp = [ref(x, mask, fill) for x, mask, fill in vecs]
            bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
            status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
            ok_all &= status == "OK"
            print(f"vec_select N={n:2d} W={w}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
            for v, g, e in bad[:3]:
                print("   mask", bin(v[1]), "fill", v[2], "rtl", g[:4], "ref", e[:4])
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
