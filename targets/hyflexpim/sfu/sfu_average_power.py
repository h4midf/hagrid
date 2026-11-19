"""Average power of one HyFlexPIM SFU under the paper's own encoder workload: the
SFU's leakage plus the energy of the operations it executes per layer, spread over
the time its PU takes for that layer. Idle units are clock-gated (no clock energy when
they have no operation; as TransPIM, by a HyFlexPIM author, states for its units), and
leakage is always on. Energies per operation: results/sfu_power.json (power_sfu.py);
leakage: results/sfu.json. Writes results/sfu_average_power.json.

Workload (HyFlexPIM Sec. 5.1, 5.4): BERT-Large, sequence length N = 128 (GLUE MSL),
24 layers, one PU per layer, hidden 1024, 16 heads, FFN 4096; INT8 activations; 8
digital modules, each with one SFU, per PU (Sec. 3.1). Per layer and sequence, the
SFUs of the PU compute (our decomposition into the SFU's functions):
  softmax   16 heads x N x N scores: max, sub (x - max), exp, add (row sum), div
  layernorm 2 x N x 1024 values: add (sum), sub (x - mean), mul (square), add (sum of
            squares), div (normalize), mul (gamma), add (beta); per row sqrt and div
            (variance), negligible next to the per-value operations
  GELU      N x 4096 values: x / (1 + exp(-1.702 x)): mul, exp, add, div
  every result leaves converted to int8 (Sec. 3.1): to_int8
Layer time: the analog linear layers bound the PU (Sec. 5.4: one input bit per 100 ns
array cycle, 8-bit inputs, one token per matrix at a time, no weight replication for
BERT-Large, which uses one PU per layer): N x 8 x 100 ns per layer and sequence, with
the PU's matrices working on different tokens at once (pipelined).

Usage (from the repo root, after power_sfu.py):
  python3 targets/hyflexpim/sfu/sfu_average_power.py
"""
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DESIGN = json.load(open(os.path.join(HERE, "..", "default", "design.json")))

N = 128                   # GLUE maximum sequence length (Sec. 5.1)
HIDDEN, HEADS, FFN = 1024, 16, 4096   # BERT-Large
INPUT_BITS = 8            # INT8 activations (Sec. 5.1)


def main():
    pw = json.load(open(os.path.join(HERE, "results", "sfu_power.json")))
    e = {fn: d["energy_pJ_per_op_65nm"] for fn, d in pw["units"].items()}
    e["exp"] = pw["exp"]["energy_pJ_per_op_65nm"]
    sfus = DESIGN["processing_unit"]["digital_modules"]

    scores = HEADS * N * N
    ln_values = 2 * N * HIDDEN
    gelu_values = N * FFN
    ops = {
        "softmax": (scores, ["max", "sub", "exp", "add", "div", "to_int8"]),
        "layernorm": (ln_values, ["add", "sub", "mul", "add", "div", "mul", "add", "to_int8"]),
        "gelu": (gelu_values, ["mul", "exp", "add", "div", "to_int8"]),
    }
    energy_uJ = {name: n * sum(e[f] for f in fns) * 1e-6 for name, (n, fns) in ops.items()}
    total_uJ = sum(energy_uJ.values())

    cycle_ns = DESIGN["clock"]["analog_cycle_ns"]
    layer_us = N * INPUT_BITS * cycle_ns * 1e-3
    dyn_mW_per_sfu = total_uJ / sfus / layer_us * 1e3

    rows = json.load(open(os.path.join(HERE, "results", "sfu.json")))
    leak_mW = next(r for r in rows if r["unit"] == "sfu_lane")["sfu_65nm"]["leakage_mW"]
    busy_cycles = sum(n * (7 if "exp" in fns else 1) for n, fns in ops.values()) / sfus / 256
    out = {
        "workload": {"model": "BERT-Large", "N": N, "sfus_per_pu": sfus, "layer_time_us": layer_us},
        "energy_per_op_pJ_65nm": e,
        "energy_per_layer_uJ": energy_uJ,
        "per_sfu": {"dynamic_mW": dyn_mW_per_sfu, "leakage_mW": leak_mW, "average_mW": dyn_mW_per_sfu + leak_mW,
                    "exp_bound_busy_fraction": busy_cycles / (layer_us * 1e3)},
    }
    path = os.path.join(HERE, "results", "sfu_average_power.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    s = out["per_sfu"]
    print(f"per layer and sequence (PU, {sfus} SFUs): " + ", ".join(f"{k} {v:.2f} uJ" for k, v in energy_uJ.items())
          + f"; layer time {layer_us:.1f} us")
    print(f"one SFU: dynamic {s['dynamic_mW']:.1f} mW + leakage {s['leakage_mW']:.1f} mW = {s['average_mW']:.1f} mW "
          f"(Table 2: 138.89 mW); busy {100 * s['exp_bound_busy_fraction']:.1f}% of the time")
    print(f"wrote {path}")


if __name__ == "__main__":
    main()
