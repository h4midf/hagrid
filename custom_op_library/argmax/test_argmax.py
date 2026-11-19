"""Functional check of argmax_tree against a Python model.

Each vector is evaluated by Yosys (eval) on random and edge-case N-lane
rows, at W = 32. Exit code 0 only if every vector matches, for every N tested.
"""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "argmax_tree.v")
W = 32
MAXV = (1 << (W - 1)) - 1
MINV = -(1 << (W - 1))


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref(lanes):
    best_i, best_v = 0, lanes[0]
    for i, v in enumerate(lanes):
        if v > best_v:
            best_v, best_i = v, i
    return best_v, best_i


def vectors(n_lanes, n, rng):
    v = []
    for _ in range(n):
        v.append(tuple(rng.randint(MINV, MAXV) for _ in range(n_lanes)))
    v.append((MAXV,) * n_lanes)          # all tied at max -- must report index 0
    v.append((MINV,) * n_lanes)          # all tied at min -- must report index 0
    v.append((0,) * n_lanes)
    v.append(tuple(range(n_lanes)))       # monotonic increasing -- max at last index
    v.append(tuple(range(n_lanes, 0, -1)))  # monotonic decreasing -- max at index 0
    return v


def parse_eval(out, signal):
    got = []
    for raw in re.findall(rf"Eval result: \\{signal} = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        got.append(int(m.group(2), 2) if m else int(raw))
    return got


def run(n_lanes, vecs):
    idx_bits = int(math.log2(n_lanes))
    lines = [f"read_verilog -sv {RTL}", f"chparam -set N {n_lanes} -set W {W} argmax_tree",
              "hierarchy -top argmax_tree", "proc"]
    for lanes in vecs:
        x_flat = 0
        for i, v in enumerate(lanes):
            x_flat |= (v & ((1 << W) - 1)) << (i * W)
        lines.append(f"eval -set x_flat {n_lanes*W}'d{x_flat} -show max_val -show max_idx")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    vals = [to_signed(v, W) for v in parse_eval(out, "max_val")]
    idxs = parse_eval(out, "max_idx")
    return list(zip(vals, idxs))


def main():
    rng = random.Random(17)
    ok_all = True
    for n_lanes in (4, 8, 16, 32, 64):
        vecs = vectors(n_lanes, 100, rng)
        got = run(n_lanes, vecs)
        exp = [ref(lanes) for lanes in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"argmax_tree N={n_lanes:3d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:3]:
            print("   lanes =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
