"""Functional check of vec_add_vv against a saturating Python add. Yosys eval on
random and edge-case N-lane rows. Exit code 0 only if every vector matches, for
every N.
"""
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl")
W = 32
MAXV = (1 << (W - 1)) - 1
MINV = -(1 << (W - 1))


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def clampW(r):
    return max(MINV, min(MAXV, r))


def pack_flat(lanes):
    v = 0
    for i, lane in enumerate(lanes):
        v |= (lane & ((1 << W) - 1)) << (i * W)
    return v


def unpack_flat(v, n_lanes, bits):
    v &= (1 << bits) - 1
    return tuple(to_signed((v >> (i * W)) & ((1 << W) - 1), W) for i in range(n_lanes))


def parse_eval(out, signal):
    got = []
    for raw in re.findall(rf"Eval result: \\{signal} = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        if m:
            got.append(int(m.group(2), 2))
        else:
            got.append(int(raw))
    return got


def vectors_vv(n_lanes, n, rng):
    v = []
    for _ in range(n):
        v.append((tuple(rng.randint(MINV, MAXV) for _ in range(n_lanes)),
                   tuple(rng.randint(MINV, MAXV) for _ in range(n_lanes))))
    v.append(((MAXV,) * n_lanes, (MAXV,) * n_lanes))
    v.append(((MINV,) * n_lanes, (MINV,) * n_lanes))
    v.append(((MAXV,) * n_lanes, (1,) * n_lanes))
    v.append(((MINV,) * n_lanes, (-1,) * n_lanes))
    v.append(((0,) * n_lanes, (0,) * n_lanes))
    return v


def run_vv(n_lanes, vecs):
    lines = [f"read_verilog -sv {os.path.join(RTL, 'vec_add_vv.v')}",
              f"chparam -set N {n_lanes} -set W {W} vec_add_vv", "hierarchy -top vec_add_vv", "proc"]
    for x, y in vecs:
        lines.append(f"eval -set x_flat {n_lanes*W}'d{pack_flat(x)} -set y_flat {n_lanes*W}'d{pack_flat(y)} "
                      "-show out_flat")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    return [unpack_flat(v, n_lanes, n_lanes * W) for v in parse_eval(out, "out_flat")]


def main():
    rng = random.Random(9)
    ok_all = True
    for n_lanes in (4, 8, 16, 32, 64):
        vecs = vectors_vv(n_lanes, 50, rng)
        got = run_vv(n_lanes, vecs)
        exp = [tuple(clampW(a + b) for a, b in zip(x, y)) for x, y in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"vec_add_vv N={n_lanes:3d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:3]:
            print("  ", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
