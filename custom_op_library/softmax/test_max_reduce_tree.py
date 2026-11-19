"""Functional check of max_reduce_tree.v against Python's max(). Yosys eval at N=8
and N=512. Exit code 0 only if every vector matches."""
import os
import random
import re
import subprocess
import sys

sys.set_int_max_str_digits(100000)  # N=512 packs a 16384-bit flat vector into one decimal literal

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "max_reduce_tree.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def pack(vals, w):
    flat = 0
    for i, v in enumerate(vals):
        flat |= (v & ((1 << w) - 1)) << (i * w)
    return flat


def run(n, w, rows):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set N {n} -set W {w} max_reduce_tree",
              "hierarchy -top max_reduce_tree", "proc -norom", "flatten", "select max_reduce_tree"]
    for row in rows:
        flat = pack(row, w)
        lines.append(f"eval -set x_flat {n * w}'d{flat} -show max_val")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    for raw in re.findall(r"Eval result: \\max_val = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        if m:
            results.append(to_signed(int(m.group(2), 2), int(m.group(1))))
        else:
            results.append(to_signed(int(raw), w))
    return results


def vectors(n, count, w, rng):
    maxv, minv = (1 << (w - 1)) - 1, -(1 << (w - 1))
    rows = [[rng.randint(minv, maxv) for _ in range(n)] for _ in range(count)]
    rows += [[0] * n, [maxv] * n, [minv] * n]
    rows[0][n // 2] = maxv
    rows[1][0] = minv
    return rows


def main():
    rng = random.Random(11)
    status = "OK"
    for n in (8, 512):
        rows = vectors(n, 20, 32, rng)
        got = run(n, 32, rows)
        exp = [max(row) for row in rows]
        bad = [(r, g, e) for r, g, e in zip(rows, got, exp) if g != e]
        ok = len(got) == len(rows) and not bad
        print(f"max_reduce_tree N={n} W=32  evaluated {len(got)}/{len(rows)}  mismatches {len(bad)}  "
              f"{'OK' if ok else 'FAIL'}")
        if not ok:
            status = "FAIL"
            for r, g, e in bad[:3]:
                print("   mismatch: rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
