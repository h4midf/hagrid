"""Builds the int8 bert-base that the compiler uses: real bert-base-uncased weights,
quantized with SmoothQuant W8A8 at kernels.bert_base.SMOOTHQUANT_ALPHA
(quantize_bert_base_encoder: every matmul int8, fc2's output not requantized) on the
inputs --smooth names (kernels.bert_base.SMOOTH_GROUPS), calibrated on real WikiText-2
text, saved to kernels.bert_base.INT8_CHECKPOINTS[--smooth].

Calibration windows are MAX_SEQ_LEN tokens ([CLS] + text + [SEP]) from the WikiText-2
train split, 15% of tokens masked as in masked-LM, through BERT's real embeddings
(outside the kernel). Before quantizing, the float kernel + MLM head is checked against
Hugging Face's logits. Reports masked-token top-1 of the float and the int8 model on
the calibration windows.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_bert_base_int8 --n-calib 32 --calib-seed 1 --smooth fc1_fc2
"""
import argparse
import os

import numpy as np
import torch

from kernels.bert_base import (INT8_CHECKPOINTS, MAX_SEQ_LEN, SMOOTH_GROUPS, SMOOTHQUANT_ALPHA, BertBaseKernel,
                               quantize_bert_base_encoder)
from run_scripts.measure_accuracy.evaluate_variant_configs import load_real_weights, masked_inputs, real_embed, windows

MAX_LOGIT_DIFF = 1e-3   # float kernel vs Hugging Face, in double


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real WikiText-2 windows the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration windows")
    parser.add_argument("--smooth", required=True, choices=SMOOTH_GROUPS, help="the inputs SmoothQuant smooths")
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

    model = model.float()
    rng = np.random.default_rng(args.calib_seed)
    wins = windows("train", tok, args.n_calib, L=MAX_SEQ_LEN - 2, seed=args.calib_seed)
    calib = []
    for win in wins:
        ids, pos = masked_inputs(win, rng)
        calib.append((real_embed(hf_model, ids).float(), pos, torch.tensor(win)[pos]))

    quantized = quantize_bert_base_encoder(model, [e for e, _, _ in calib], SMOOTHQUANT_ALPHA, args.smooth)

    def masked_top1(encoder):
        correct = total = 0
        with torch.no_grad():
            for e, pos, true in calib:
                pred = torch.argmax(mlm_head(encoder(e).double())[0][pos], dim=-1)
                correct += int((pred == true).sum())
                total += len(pos)
        return correct, total

    fc, n = masked_top1(model)
    qc, _ = masked_top1(quantized)
    print(f"on the {len(calib)} calibration windows ({n} masked tokens): "
          f"float masked top-1 {fc}/{n} ({fc / n:.1%}), int8 masked top-1 {qc}/{n} ({qc / n:.1%})")

    out = INT8_CHECKPOINTS[args.smooth]
    os.makedirs(os.path.dirname(out), exist_ok=True)
    torch.save(quantized, out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
