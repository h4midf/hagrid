"""Functional check of token_scale.v against token_quant_ref.token_scale (bit-exact, by
Yosys eval), on every bit length of the row max and edge cases, at MW = 12 (the
mantissa width quant_mant_round is characterized at). Also checks that m fits MW bits,
and, in Python, that the whole per-token path (absmax, token_scale, quant_mant_round)
is within one int8 step of the ideal round(x * 127 / M). Exit code 0 only if all pass."""
import os
import random
import re
import subprocess
import sys

from token_quant_ref import MW, quantize_row, token_scale

HERE = os.path.dirname(os.path.abspath(__file__))
RTL = os.path.join(HERE, "rtl", "token_scale.v")
W = 32


def run(values):
    lines = [f"read_verilog -sv {RTL}", f"chparam -set W {W} -set MW {MW} token_scale",
             "hierarchy -top token_scale", "proc", "flatten"]
    for v in values:
        lines.append(f"eval -set absmax {W}'d{v} -show m -show sh")
    out = subprocess.run(["yosys", "-p", "; ".join(lines)], capture_output=True, text=True).stdout
    ms = [int(b, 2) for b in re.findall(r"Eval result: \\m = \d+'([01]+)", out)]
    shs = [int(b, 2) for b in re.findall(r"Eval result: \\sh = \d+'([01]+)", out)]
    return list(zip(ms, shs))


def values(rng):
    vals = [0, 1, 2, 3, 127, 128, 255, (1 << 31) - 1, 1 << 31, (1 << 32) - 1]
    for bits in range(1, W + 1):          # every bit length, its ends and random inside
        lo, hi = 1 << (bits - 1), (1 << bits) - 1
        vals += [lo, hi] + [rng.randint(lo, hi) for _ in range(4)]
    return vals


def ideal(x, m_abs):
    """round(x * 127 / M), round to nearest, ties up, as an exact rational."""
    return (2 * x * 127 + m_abs) // (2 * m_abs) if m_abs else 0


def main():
    rng = random.Random(17)
    vals = values(rng)
    got = run(vals)
    exp = [token_scale(v) for v in vals]
    bad = [(v, g, e) for v, g, e in zip(vals, got, exp) if g != e]
    wide = [(v, g) for v, g in zip(vals, got) if g[0] >= (1 << MW)]
    ok = len(got) == len(vals) and not bad and not wide
    print(f"token_scale W={W} MW={MW}  evaluated {len(got)}/{len(vals)}  mismatches {len(bad)}  "
          f"m >= 2^MW: {len(wide)}  {'OK' if ok else 'FAIL'}")
    for v, g, e in bad[:3]:
        print("   M =", v, "rtl (m, sh)", g, "ref", e)

    # whole path vs ideal, on activation-like rows (Q12.20, magnitudes up to 2048)
    worst = 0
    for _ in range(300):
        scale = 1 << rng.randint(0, 31)
        row = [rng.randint(-scale, scale) for _ in range(768)]
        q, m_abs = quantize_row(row)
        worst = max(worst, max(abs(a - ideal(x, m_abs)) for a, x in zip(q, row)))
    path_ok = worst <= 1
    print(f"per-token path vs round(x * 127 / M): worst error {worst} int8 step(s)  {'OK' if path_ok else 'FAIL'}")
    sys.exit(0 if ok and path_ok else 1)


if __name__ == "__main__":
    main()
