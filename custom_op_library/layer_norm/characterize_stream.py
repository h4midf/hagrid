"""Synthesises one streamed layer-norm circuit at a row of --row elements, N lanes per
cycle (4 and 8), at both characterized widths, and writes its rows into its library's
results file (the other circuits' rows are kept):

  unit                      top RTL                               results
  layer_norm_stream         rtl/layer_norm_stream.v               layer_norm/results/layer_norm.json
  ailayernorm_stream        rtl/ailayernorm_stream.v              layer_norm/results/layer_norm.json

Hierarchical synthesis (synth.synth_hier): ABC maps every module on its own, since the
flat netlists do not map in time. Area counts every instance. The clock period
(delay_ps) is the longest register-to-register path, composed from the module delays:
each unit lists its paths as chains of modules ("top" is the unit's own logic, the
registers, accumulators and muxes), a chain's delay is the sum of its modules' ABC
critical paths, and delay_ps is the largest chain. Nothing is optimized or timed across
a module boundary, so this is an upper bound. cycles_per_row is the cycles one row
takes. A config whose synthesis exceeds synth.DEFAULT_TIMEOUT_S is killed and dropped
(no row). Functional correctness: the matching test_*_stream.py.

Usage (from the repo root):
  python3 custom_op_library/layer_norm/characterize_stream.py --unit layer_norm_stream --row 1024
Replaces the unit's rows at --row in its results file; its rows at other row lengths stay.
"""
import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common", "tools"))
import report  # noqa: E402
import synth  # noqa: E402

LANES = (4, 8)
WIDTHS = ((32, 20, "Q12.20"), (16, 8, "Q8.8"))
TIMEOUT_MESSAGE = "did not complete within"   # synth's message when it kills a run
RSQRT = "rsqrt"                                # stands for rsqrt_q12_20 / rsqrt_q8_8 in a chain


def _ln(*names):
    return [os.path.join(HERE, "rtl", n) for n in names]


# unit -> (RTL files, library dir, results op, method, cycles_per_row(n, w, frac),
#          register-to-register chains)
UNITS = {
    "layer_norm_stream": (
        _ln("reduce_tree.v", "rsqrt_q12_20.v", "rsqrt_q8_8.v", "row_div.v", "ln_stream_lane.v",
            "layer_norm_stream.v"), HERE, "layer_norm",
        "streamed layer_norm.v: 3 passes of N lanes (sum, squared differences, normalize + "
        "per-element affine), rsqrt (1 Newton iteration) once per row",
        lambda n, w, frac, row: 3 * row // n + 1,
        {"SUM/VAR": ["ln_stream_lane", "reduce_tree", "top", "row_div"],
         "RSQ": [RSQRT, "top"],
         "NORM": ["ln_stream_lane"]}),
    "ailayernorm_stream": (
        _ln("reduce_tree.v", "rsqrt_q12_20.v", "rsqrt_q8_8.v", "row_div.v", "ailn_stream_lane.v", "ailn_var.v",
            "ailayernorm_stream.v"), HERE, "layer_norm",
        "streamed SOLE AILayerNorm (ICCAD 2023): 2 passes of N lanes (PTF quantize + "
        "DynamicCompress statistics, normalize + per-element affine), rsqrt once per row",
        lambda n, w, frac, row: 2 * row // n + 1,
        {"STATS": ["ailn_stream_lane", "reduce_tree", "top", "row_div"],
         "RSQ": ["ailn_var", RSQRT, "top"],
         "NORM": ["ailn_stream_lane"]}),
}


def clock_ps(unit, w, delays, chains):
    """The largest chain: the sum of its modules' ABC critical paths."""
    rsqrt = "rsqrt_q12_20" if w == 32 else "rsqrt_q8_8"
    per_chain = {}
    for name, chain in chains.items():
        mods = [unit if m == "top" else rsqrt if m == RSQRT else m for m in chain]
        missing = [m for m in mods if m not in delays]
        if missing:
            raise KeyError(f"{unit}: no ABC delay for {missing} (have {sorted(delays)})")
        per_chain[name] = round(sum(delays[m] for m in mods), 2)
    return max(per_chain.values()), per_chain


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--unit", required=True, choices=sorted(UNITS))
    parser.add_argument("--row", type=int, required=True, help="row length (a model's hidden size)")
    args = parser.parse_args()
    files, lib_dir, op, method, cycles, chains = UNITS[args.unit]
    # the unit's rows at other row lengths are kept; its rows at --row are replaced
    with open(os.path.join(lib_dir, "results", f"{op}.json")) as f:
        rows = [r for r in json.load(f) if r["unit"] == args.unit and r["params"]["ROW"] != args.row]
    for w, frac, fmt in WIDTHS:
        for n in LANES:
            params = {"N": n, "W": w, "FRAC": frac, "ROW": args.row}
            try:
                res = synth.synth_hier(files, args.unit, params)
            except RuntimeError as err:
                # synth kills a run that exceeds its timeout: that config is dropped
                if TIMEOUT_MESSAGE not in str(err):
                    raise
                print(f"  {args.unit} N={n} W={w}: DROPPED, synthesis did not complete within "
                      f"{synth.DEFAULT_TIMEOUT_S}s", flush=True)
                continue
            clock, per_chain = clock_ps(args.unit, w, res["module_delays_ps"], chains)
            r = report.row(args.unit, {**params, "in": fmt, "out": fmt}, method,
                           {"area_um2": res["area_um2"], "delay_ps": clock, "cells": res["cells"]},
                           f"bit-exact vs its Python model (test_{args.unit}.py); hierarchical synthesis, "
                           "delay_ps is the clock period: the largest register-to-register chain of "
                           "module delays (an upper bound)")
            r["cycles_per_row"] = cycles(n, w, frac, args.row)
            r["module_delays_ps"] = res["module_delays_ps"]
            r["chain_delays_ps"] = per_chain
            rows.append(r)
            print(f"  {args.unit} N={n} W={w}: area {res['area_um2']:.0f} um^2, clock {clock} ps "
                  f"{per_chain}, {r['cycles_per_row']} cycles/row", flush=True)
    report.save_unit(lib_dir, op, args.unit, rows)


if __name__ == "__main__":
    main()
