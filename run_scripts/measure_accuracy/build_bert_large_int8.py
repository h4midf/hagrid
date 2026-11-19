"""Builds the int8 bert-large that the compiler uses: real bert-large-uncased weights,
quantized with kernels/bert_large.py's scheme (quantize_bert_encoder), calibrated on
real WikiText-2 text, saved to kernels.bert_large.INT8_CHECKPOINT.

Calibration windows are MAX_SEQ_LEN tokens ([CLS] + text + [SEP]) from the WikiText-2
train split, 15% of tokens masked as in masked-LM, through BERT's real embeddings
(outside the kernel).
Before quantizing, the float kernel + MLM head is checked against Hugging Face's
logits. Reports masked-token top-1 of the float and the int8 model on the
calibration windows.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_bert_large_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import os

import numpy as np
import torch

from kernels.bert_large import INT8_CHECKPOINT, MAX_SEQ_LEN, BertLargeKernel, quantize_bert_encoder
from run_scripts.measure_accuracy.evaluate_bert_large_variant_configs import load_real_weights
from run_scripts.measure_accuracy.evaluate_variant_configs import masked_inputs, real_embed, windows

MAX_LOGIT_DIFF = 1e-3   # float kernel vs Hugging Face, in double


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real WikiText-2 windows the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration windows")
    args = parser.parse_args()

    model = BertLargeKernel().eval().double()
    tok, hf_model, mlm_head = load_real_weights(model)

    test_win = windows("test", tok, 1, L=MAX_SEQ_LEN - 2, seed=99)[0]
    with torch.no_grad():
        mine = mlm_head(model(real_embed(hf_model, np.array(test_win))))[0]
        ref = hf_model(input_ids=torch.tensor([test_win])).logits[0]
    max_diff = float((mine - ref).abs().max())
    print(f"check vs HF logits ({len(test_win)} tokens): max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"BertLargeKernel + MLM head does not match Hugging Face bert-large (diff {max_diff})")

    model = model.float()
    rng = np.random.default_rng(args.calib_seed)
    wins = windows("train", tok, args.n_calib, L=MAX_SEQ_LEN - 2, seed=args.calib_seed)
    calib = []
    for win in wins:
        ids, pos = masked_inputs(win, rng)
        calib.append((real_embed(hf_model, ids).float(), pos, torch.tensor(win)[pos]))

    quantized = quantize_bert_encoder(model, [e for e, _, _ in calib])

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

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized, INT8_CHECKPOINT)
    print(f"wrote {INT8_CHECKPOINT}")


if __name__ == "__main__":
    main()
