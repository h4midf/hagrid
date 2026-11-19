"""Merges the sweep's shards (bert_base_quant_sweep.py) and the plan
(bert_base_variant_plan.py) into one JSON with every variant of
variant_generators/bert_base_quant.py:

  {"float_ppl", "int8_torch_ppl", "n_eval", "seed", "calib_windows", "calib_seed",
   "variants": {name: {"model": kernel, "flags": the complete flags,
                       and one of
                       "ppl"                         its evaluation,
                       "ppl" + "same_computation_as" the evaluated variant it computes the
                                                     same values as,
                       "unrepresentable"             why a circuit cannot encode a scale}}}

model and flags are what compiler.lowering.build_tiled_cim_block(model, flags) compiles.
Raises if a planned group has no evaluation in any shard, or if the shards disagree on
the baselines or settings.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.bert_base_variants_merge \
      --plan run_scripts/results/bert_base_variants_plan.json \
      --shards run_scripts/results/bert_base_variants_ppl/shard_*.json \
      --out run_scripts/results/bert_base_variants.json
"""
import argparse
import json

from variant_generators.bert_base_quant import MODEL_OF_SMOOTH, all_variants

_SHARED = ("float_ppl", "int8_torch_ppl", "n_eval", "seed", "calib_windows", "calib_seed", "prep", "plan")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--shards", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    with open(args.plan) as f:
        plan = json.load(f)
    evaluated, shared = {}, None
    for path in args.shards:
        with open(path) as f:
            shard = json.load(f)
        settings = {k: shard[k] for k in _SHARED}
        if shared is None:
            shared = settings
        elif settings != shared:
            raise ValueError(f"{path}: settings {settings} differ from the other shards' {shared}")
        evaluated.update(shard["variants"])

    flags_of = dict(all_variants())
    variants = {}
    missing = []
    for group in plan["groups"]:
        head = group[0]
        if head not in evaluated:
            missing.append(head)
            continue
        for name in group:
            entry = {"model": MODEL_OF_SMOOTH[flags_of[name]["smooth"]], "flags": flags_of[name],
                     "ppl": evaluated[head]["ppl"]}
            if name != head:
                entry["same_computation_as"] = head
            variants[name] = entry
    if missing:
        raise ValueError(f"{len(missing)} planned groups have no evaluation, e.g. {missing[:3]}")
    for name, reason in plan["unrepresentable"].items():
        variants[name] = {"model": MODEL_OF_SMOOTH[flags_of[name]["smooth"]], "flags": flags_of[name],
                          "unrepresentable": reason}
    if len(variants) != len(flags_of):
        raise ValueError(f"{len(variants)} variants merged, the generator has {len(flags_of)}")

    with open(args.out, "w") as f:
        json.dump({**shared, "variants": variants}, f, indent=1)
    print(f"wrote {args.out}: {len(variants)} variants")


if __name__ == "__main__":
    main()
