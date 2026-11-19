"""Groups the variants of variant_generators/bert_base_quant.py by what the hardware
computes, so each distinct computation is evaluated once
(run_scripts/measure_accuracy/bert_base_quant_sweep.py --plan).

Two variants compute the same values when they run the same checkpoint and their
emulators (bert_base_int8_emulator.Int8Emulator) have equal numeric signatures: the
data width, every quantize / dequantize / requantize encoding (e.g. two mantissa widths
whose encodings of every op's scale are the same (m, sh)), each x / s's form and every
op's circuit. A variant whose circuit cannot represent an op's scale is recorded as
unrepresentable. No inference is run.

Writes {"groups": [[variant names] per distinct computation, the first is the one
evaluated], "unrepresentable": {name: reason}}.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.bert_base_variant_plan --out run_scripts/results/bert_base_variants_plan.json
"""
import argparse
import json
import time

from torch.fx import symbolic_trace

from kernels.registry import get_kernel
from run_scripts.measure_accuracy.bert_base_int8_emulator import Int8Emulator
from run_scripts.measure_accuracy.bert_base_quant_sweep import UNREPRESENTABLE
from variant_generators import common
from variant_generators.bert_base_quant import MODEL_OF_SMOOTH, all_variants

# SOLE's layer norm needs PTF exponents to build its emulator; they are the same for
# every variant, so a placeholder is enough to compare encodings
_PTF_PLACEHOLDER = 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", required=True, help="plan JSON")
    args = parser.parse_args()

    traced, discovered = {}, {}
    for smooth, kernel_name in MODEL_OF_SMOOTH.items():
        kernel_cls, _, _ = get_kernel(kernel_name)
        traced[smooth] = symbolic_trace(kernel_cls())
        discovered[smooth] = common.discover_alt_ops(traced[smooth])

    groups: dict[tuple, list[str]] = {}
    unrepresentable = {}
    t0 = time.time()
    for i, (name, flags) in enumerate(all_variants()):
        smooth = flags["smooth"]
        alt_ops, quant_data = discovered[smooth]
        try:
            config = common.build_config_for_variant(alt_ops, flags, quant_data)
            for cfg in config.values():
                if cfg["kind"] == "layer_norm" and cfg["impl"] == "sole":
                    cfg["ptf_alpha"] = [_PTF_PLACEHOLDER] * quant_data_hidden(quant_data)
            emulator = Int8Emulator(traced[smooth], config, flags["data_width"], count_saturation=False)
        except ValueError as err:
            if UNREPRESENTABLE not in str(err):
                raise
            unrepresentable[name] = str(err)
            continue
        groups.setdefault((smooth, emulator.numeric_signature()), []).append(name)
        if i % 500 == 0:
            print(f"  {i} variants, {len(groups)} distinct, {len(unrepresentable)} unrepresentable "
                  f"({time.time() - t0:.0f}s)", flush=True)

    plan = {"groups": list(groups.values()), "unrepresentable": unrepresentable}
    with open(args.out, "w") as f:
        json.dump(plan, f, indent=1)
    print(f"{sum(len(g) for g in groups.values())} representable variants in {len(groups)} distinct computations, "
          f"{len(unrepresentable)} unrepresentable; wrote {args.out}")


def quant_data_hidden(quant_data: dict) -> int:
    """The layer norms' hidden size (one per model; raises if they differ)."""
    sizes = {v["hidden_size"] for v in quant_data.values() if "hidden_size" in v}
    if len(sizes) != 1:
        raise ValueError(f"layer norm hidden sizes {sizes}")
    return sizes.pop()


if __name__ == "__main__":
    main()
