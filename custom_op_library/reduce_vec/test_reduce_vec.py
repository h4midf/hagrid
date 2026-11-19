"""Functional check of reduce_tree: Yosys eval on random and edge-case N-lane rows
against Python's sum. Exit code 0 only if every vector matches, for every N."""
import math
import os
import random
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "reduce_tree.v")
W = 32


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def ref(lanes):
    return sum(lanes)


def vectors(n_lanes, n, rng):
    v = []
    for _ in range(n):
        v.append(tuple(rng.randint(-(1 << (W - 1)), (1 << (W - 1)) - 1) for _ in range(n_lanes)))
    # edge cases: all max, all min, all zero, alternating max/min
    v.append(((1 << (W - 1)) - 1,) * n_lanes)
    v.append((-(1 << (W - 1)),) * n_lanes)
    v.append((0,) * n_lanes)
    v.append(tuple(((1 << (W - 1)) - 1) if i % 2 == 0 else -(1 << (W - 1)) for i in range(n_lanes)))
    return v


def run(n_lanes, vecs):
    out_bits = W + int(math.log2(n_lanes))
    lines = [f"read_verilog -sv {RTL}", f"chparam -set N {n_lanes} -set W {W} reduce_tree",
              "hierarchy -top reduce_tree", "proc"]
    for lanes in vecs:
        x_flat = 0
        for i, v in enumerate(lanes):
            x_flat |= (v & ((1 << W) - 1)) << (i * W)
        lines.append(f"eval -set x_flat {n_lanes * W}'d{x_flat} -show sum")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    got = []
    for raw in re.findall(r"Eval result: \\sum = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        if m:
            bits, bitstr = int(m.group(1)), m.group(2)
            got.append(to_signed(int(bitstr, 2), bits))
        else:
            got.append(int(raw))
    return got


def main():
    rng = random.Random(5)
    ok_all = True
    for n_lanes in (4, 8, 16, 32, 64):
        vecs = vectors(n_lanes, 150, rng)
        got = run(n_lanes, vecs)
        exp = [ref(lanes) for lanes in vecs]
        bad = [(v, g, e) for v, g, e in zip(vecs, got, exp) if g != e]
        status = "OK" if len(got) == len(vecs) and not bad else "FAIL"
        ok_all &= status == "OK"
        print(f"reduce_tree N={n_lanes:3d}  evaluated {len(got)}/{len(vecs)}  mismatches {len(bad)}  {status}")
        for v, g, e in bad[:3]:
            print("   lanes =", v, "rtl", g, "ref", e)
    sys.exit(0 if ok_all else 1)


if __name__ == "__main__":
    main()
