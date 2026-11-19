"""Functional check of log2exp_standalone.v(d << 8) against e2softmax_row8.v's
log2exp_q8_8 model, for every d in -255..0. Exit code 0 only if every vector
matches."""
import os
import re
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "log2exp_standalone.v")


def to_signed(v, bits):
    v &= (1 << bits) - 1
    if v >= (1 << (bits - 1)):
        v -= (1 << bits)
    return v


def log2exp_fixed_original(d):
    """Already-validated reference, see test_e2softmax_row8.py's own copy."""
    x = d << 8
    xshift = x + (x >> 1) - (x >> 4)
    k_raw = xshift >> 8
    k = -k_raw
    return max(0, min(15, k))


def recon(k):
    return 0 if k > 8 else (256 >> k)


def run(xs):
    lines = [f"read_verilog -sv {RTL}", "hierarchy -top log2exp_standalone", "proc -norom", "flatten",
             "select log2exp_standalone"]
    for x in xs:
        lines.append(f"eval -set x 24'd{x & 0xFFFFFF} -show k_y -show e_recon")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    results = []
    cur = {}
    for name, raw in re.findall(r"Eval result: \\(k_y|e_recon) = ([^\n]+)\.", out):
        m = re.match(r"(\d+)'([01]+)$", raw)
        cur[name] = int(m.group(2), 2) if m else int(raw)
        if len(cur) == 2:
            results.append((cur["k_y"], cur["e_recon"]))
            cur = {}
    return results


def main():
    ds = list(range(-255, 1))
    xs = [d << 8 for d in ds]
    got = run(xs)
    exp = [(log2exp_fixed_original(d), recon(log2exp_fixed_original(d))) for d in ds]
    bad = [(d, g, e) for d, g, e in zip(ds, got, exp) if g != e]
    status = "OK" if len(got) == len(ds) and not bad else "FAIL"
    print(f"log2exp_standalone  evaluated {len(got)}/{len(ds)}  mismatches {len(bad)}  {status}")
    for d, g, e in bad[:5]:
        print("   d =", d, "rtl", g, "ref", e)
    sys.exit(0 if status == "OK" else 1)


if __name__ == "__main__":
    main()
