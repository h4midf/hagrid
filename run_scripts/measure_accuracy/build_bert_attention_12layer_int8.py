"""Builds the int8 12-layer attention app the compiler maps (kernels/bert_attention_only.py,
the paper's mapping motivation): bert-base-uncased's real attention weights of its 12
layers, stacked without FFN / LayerNorm, quantized with bert-base's scheme
(quantize_attention_stack), calibrated on real WikiText-2 text, saved to
kernels.bert_attention_only.INT8_CHECKPOINT.

The calibration inputs are the same as build_bert_base_int8.py's: MAX_SEQ_LEN-token
windows ([CLS] + text + [SEP]) from the WikiText-2 train split, 15% of tokens masked,
through BERT's real embeddings. Before quantizing, the float bert-base kernel + MLM head is
checked against Hugging Face's logits (the weights it takes the attention from). Reports
the int8 stack's output error against the float stack on the calibration windows.

With --layers 1 it builds the same app with only bert-base's first attention layer (the
paper's mapping example, Fig. 6), saved to INT8_1LAYER_CHECKPOINT: equal to the 12-layer
app's first layer, whose calibration input is also the embeddings.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_bert_attention_12layer_int8 --layers 12 --n-calib 64 --calib-seed 1
  python3 -m run_scripts.measure_accuracy.build_bert_attention_12layer_int8 --layers 1 --n-calib 64 --calib-seed 1
"""
import argparse
import copy
import os

import numpy as np
import torch

from kernels.bert_attention_only import (
    INT8_1LAYER_CHECKPOINT, INT8_CHECKPOINT, NUM_LAYERS, AttentionStack, quantize_attention_stack,
)
from kernels.bert_base import MAX_SEQ_LEN, SMOOTHQUANT_ALPHA, BertBaseKernel
from run_scripts.measure_accuracy.evaluate_variant_configs import load_real_weights, masked_inputs, real_embed, windows

MAX_LOGIT_DIFF = 1e-3   # float kernel vs Hugging Face, in double


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--layers", type=int, required=True, choices=(1, NUM_LAYERS),
                        help="bert-base's first attention layer only, or all of them")
    parser.add_argument("--n-calib", type=int, required=True, help="real WikiText-2 windows the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration windows")
    args = parser.parse_args()

    model = BertBaseKernel().eval().double()
    tok, hf_model, mlm_head = load_real_weights(model)
    test_win = windows("test", tok, 1, L=MAX_SEQ_LEN - 2, seed=99)[0]
    with torch.no_grad():
        mine = mlm_head(model(real_embed(hf_model, np.array(test_win))))[0]
        ref = hf_model(input_ids=torch.tensor([test_win])).logits[0]
    max_diff = float((mine - ref).abs().max())
    print(f"check vs HF logits ({len(test_win)} tokens): max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"BertBaseKernel + MLM head does not match Hugging Face bert-base (diff {max_diff})")

    stack = AttentionStack([copy.deepcopy(block.attn) for block in model.layers[:args.layers]]).eval().float()
    rng = np.random.default_rng(args.calib_seed)
    wins = windows("train", tok, args.n_calib, L=MAX_SEQ_LEN - 2, seed=args.calib_seed)
    calib = [real_embed(hf_model, masked_inputs(win, rng)[0]).float() for win in wins]

    quantized = quantize_attention_stack(stack, calib, SMOOTHQUANT_ALPHA)

    with torch.no_grad():
        num = den = 0.0
        for x in calib:
            ref_out, q_out = stack(x), quantized(x)
            num += float((q_out - ref_out).pow(2).sum())
            den += float(ref_out.pow(2).sum())
    print(f"on the {len(calib)} calibration windows: int8 vs float stack output, "
          f"relative RMS error {(num / den) ** 0.5:.3%}")

    out = INT8_CHECKPOINT if args.layers == NUM_LAYERS else INT8_1LAYER_CHECKPOINT
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(quantized, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
