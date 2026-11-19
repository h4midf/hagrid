"""Masked-LM pseudo-perplexity of bert-base for the variants of
variant_generators/bert_base_quant.py, each run on its int8 checkpoint through the
hardware emulator (run_scripts/measure_accuracy/bert_base_int8_emulator.py), once per
distinct computation: bert_base_variant_plan.py's groups (--plan), the first variant of
each group evaluated, its entry listing the others (same_computation); the groups are
split over --num-shards processes. The evaluation samples, the MLM head, SOLE's PTF
exponents and the baselines come from bert_base_sweep_prep.py's file (--prep).

Resumable: a group already evaluated in any results file of --out's directory (same
prep and plan, any earlier shard count) is skipped, and --out keeps its earlier results.

Each entry has the kernel and the complete flags, enough to compile the variant
(compiler.lowering.build_tiled_cim_block(model, flags)). bert_base_variants_merge.py
merges the shards and the plan into one file.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.bert_base_quant_sweep --prep dump/bert_base_sweep_prep.pt \
      --plan run_scripts/results/bert_base_variants_plan.json \
      --shard 0 --num-shards 5 --out run_scripts/results/bert_base_variants_ppl/shard_0.json
Writes --out after every run.
"""
import argparse
import glob
import json
import os
import time

import torch
from torch.fx import symbolic_trace

from kernels.registry import get_kernel
from run_scripts.measure_accuracy.bert_base_int8_emulator import Int8Emulator
from run_scripts.measure_accuracy.bert_base_sweep_prep import load_mlm_head, pseudo_perplexity
from variant_generators import common
from variant_generators.bert_base_quant import MODEL_OF_SMOOTH, all_variants

UNREPRESENTABLE = "no (m, sh)"   # quant_encoding.encode_mantissa's message for a scale it cannot encode
# the prep file's settings and baselines, carried into every shard's output
PREP_FIELDS = ("n_eval", "seed", "calib_windows", "calib_seed", "float_ppl", "int8_torch_ppl")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--prep", required=True, help="bert_base_sweep_prep.py's output")
    parser.add_argument("--plan", required=True, help="bert_base_variant_plan.py's output")
    parser.add_argument("--shard", type=int, required=True, help="this process runs the groups g with "
                        "g %% --num-shards == --shard")
    parser.add_argument("--num-shards", type=int, required=True, help="processes the groups are split over")
    parser.add_argument("--out", required=True, help="results JSON of this shard")
    args = parser.parse_args()
    if not 0 <= args.shard < args.num_shards:
        raise ValueError(f"--shard {args.shard} not in [0, {args.num_shards})")

    prep = torch.load(args.prep, weights_only=False)
    samples, mlm_head, ptf_alpha = prep["samples"], load_mlm_head(prep), prep["ptf_alpha"]
    traced, discovered = {}, {}
    for smooth, kernel_name in MODEL_OF_SMOOTH.items():
        kernel_cls, _, _ = get_kernel(kernel_name)
        traced[smooth] = symbolic_trace(kernel_cls())
        discovered[smooth] = common.discover_alt_ops(traced[smooth])
    with open(args.plan) as f:
        plan = json.load(f)
    flags_of = dict(all_variants())

    result = {**{k: prep[k] for k in PREP_FIELDS}, "prep": args.prep, "plan": args.plan, "shard": args.shard,
              "num_shards": args.num_shards, "variants": {}}
    # resumable: every results file in --out's directory (any earlier run, any shard
    # count) with this prep and plan; a group evaluated in one of them is skipped, and
    # --out keeps the results it already has
    done = set()
    for path in sorted(glob.glob(os.path.join(os.path.dirname(os.path.abspath(args.out)), "*.json"))):
        with open(path) as f:
            earlier = json.load(f)
        if (earlier["prep"], earlier["plan"]) != (args.prep, args.plan):
            raise ValueError(f"{path} was written with prep {earlier['prep']!r} and plan {earlier['plan']!r}, "
                             f"not {args.prep!r} and {args.plan!r}: use another output directory")
        done.update(earlier["variants"])
        if os.path.abspath(path) == os.path.abspath(args.out):
            result["variants"].update(earlier["variants"])
    print(f"{len(done)} groups already evaluated in {os.path.dirname(args.out)}", flush=True)
    for g, group in enumerate(plan["groups"]):
        if g % args.num_shards != args.shard:
            continue
        name = group[0]
        if name in done:
            continue
        flags = flags_of[name]
        smooth = flags["smooth"]
        alt_ops, quant_data = discovered[smooth]
        t0 = time.time()
        config = common.build_config_for_variant(alt_ops, flags, quant_data)
        if flags["ln_impl"] == "sole":
            for fx_name, cfg in config.items():
                if cfg["kind"] == "layer_norm":
                    cfg["ptf_alpha"] = ptf_alpha[fx_name]
        emulator = Int8Emulator(traced[smooth], config, flags["data_width"], count_saturation=False)
        # the kernel and the complete flags: enough to compile the variant
        entry = {"model": MODEL_OF_SMOOTH[smooth], "flags": flags, "same_computation": group[1:],
                 "ppl": pseudo_perplexity(emulator.forward_real, mlm_head, samples)}
        print(f"{name} (+{len(group) - 1} same): {entry['ppl']:.4f} ({time.time() - t0:.0f}s)", flush=True)
        result["variants"][name] = entry
        with open(args.out, "w") as f:
            json.dump(result, f, indent=2)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
