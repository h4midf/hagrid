"""The one-time work of bert-base's variant sweep (bert_base_quant_sweep.py), saved once
so every shard only loads it:

  - real bert-base-uncased weights (Hugging Face), the float kernel checked against
    Hugging Face's logits first;
  - the evaluation samples: --n-eval WikiText-2 test windows of MAX_SEQ_LEN tokens
    ([CLS] + text + [SEP]), 15% of the tokens masked (seed --seed), each embedded by
    BERT's real embeddings (outside the kernel), with its masked positions and true
    tokens;
  - the MLM head's weights (BERT's real head, outside the kernel);
  - SOLE's per-channel PTF exponents of every layer norm (calibrate_ptf_alpha, on
    --calib-windows train windows, seed --calib-seed);
  - the baselines: the float kernel's pseudo-perplexity, and torch's own int8
    pseudo-perplexity of each SmoothQuant checkpoint (variant_generators/
    bert_base_quant.py MODEL_OF_SMOOTH).

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.bert_base_sweep_prep --n-eval 32 --seed 3 \
      --calib-windows 32 --calib-seed 1 --out dump/bert_base_sweep_prep.pt
"""
import argparse
import math
import time

import numpy as np
import torch

from kernels.bert_base import MAX_SEQ_LEN, BertBaseKernel
from kernels.registry import get_kernel
from run_scripts.measure_accuracy.evaluate_variant_configs import (
    MLMHead, load_real_weights, masked_inputs, real_embed, windows,
)
from variant_generators.bert_base_quant import MODEL_OF_SMOOTH

MAX_LOGIT_DIFF = 1e-3   # float kernel vs Hugging Face, in double


def pseudo_perplexity(encoder, mlm_head, samples: list) -> float:
    """exp(mean NLL of the true token at every masked position); `encoder` maps a real
    [1, T, DIM] input to real [1, T, DIM] hidden states."""
    nll, count = 0.0, 0
    with torch.no_grad():
        for e, pos, true in samples:
            lp = torch.log_softmax(mlm_head(encoder(e).double())[0], dim=-1)
            nll += -lp[pos, true].sum().item()
            count += len(pos)
    return math.exp(nll / count)


def calibrate_ptf_alpha(model, tok, hf_model, n_windows: int, seed: int) -> dict:
    """SOLE's PTF exponents ({layer_norm fx_name: [alpha per channel]}): per channel,
    ceil(log2(max |x| / 127)) of each layer norm's input over `n_windows` WikiText-2
    train windows through the float model (run_scripts/measure_accuracy/
    sole_bert_large.py ptf_quantize), clamped to the circuit's 8-bit signed alpha."""
    amax = {}

    def record(key):
        def hook(_module, inputs):
            v = inputs[0].detach().abs().reshape(-1, inputs[0].shape[-1]).amax(0)
            amax[key] = torch.maximum(amax[key], v) if key in amax else v
        return hook

    handles = []
    for layer, block in enumerate(model.layers):
        handles.append(block.ln1.register_forward_pre_hook(record(f"_quantized_layers_{layer}_ln1")))
        handles.append(block.ln2.register_forward_pre_hook(record(f"_quantized_layers_{layer}_ln2")))
    rng = np.random.default_rng(seed)
    with torch.no_grad():
        for win in windows("train", tok, n_windows, L=MAX_SEQ_LEN - 2, seed=seed):
            ids, _ = masked_inputs(win, rng)
            model(real_embed(hf_model, ids))
    for h in handles:
        h.remove()
    return {k: torch.clamp(torch.ceil(torch.log2(v.clamp(min=1e-12) / 127.0)), -128, 127).long().tolist()
            for k, v in amax.items()}


def load_mlm_head(prep: dict) -> MLMHead:
    """The MLM head saved in a prep file, in double."""
    head = MLMHead(prep["mlm_head_dim"], prep["mlm_head_vocab"]).eval().double()
    head.load_state_dict(prep["mlm_head"])
    return head


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-eval", type=int, required=True, help="WikiText-2 test windows")
    parser.add_argument("--seed", type=int, required=True, help="test window shuffle and mask seed")
    parser.add_argument("--calib-windows", type=int, required=True,
                        help="WikiText-2 train windows SOLE's PTF exponents are calibrated on")
    parser.add_argument("--calib-seed", type=int, required=True, help="calibration window shuffle and mask seed")
    parser.add_argument("--out", required=True, help="prep file (torch.save)")
    args = parser.parse_args()

    model = BertBaseKernel().eval().double()
    tok, hf_model, mlm_head = load_real_weights(model)
    test_win = windows("test", tok, 1, L=MAX_SEQ_LEN - 2, seed=99)[0]
    with torch.no_grad():
        mine = mlm_head(model(real_embed(hf_model, np.array(test_win))))[0]
        ref = hf_model(input_ids=torch.tensor([test_win])).logits[0]
    max_diff = float((mine - ref).abs().max())
    print(f"check vs HF logits: max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"BertBaseKernel + MLM head does not match Hugging Face bert-base (diff {max_diff})")

    rng = np.random.default_rng(args.seed)
    samples = []
    for win in windows("test", tok, args.n_eval, L=MAX_SEQ_LEN - 2, seed=args.seed):
        ids, pos = masked_inputs(win, rng)
        samples.append((real_embed(hf_model, ids), pos, torch.tensor(win)[pos]))

    t0 = time.time()
    float_ppl = pseudo_perplexity(model, mlm_head, samples)
    print(f"float: {float_ppl:.4f} ({time.time() - t0:.0f}s)", flush=True)
    int8_torch_ppl = {}
    for smooth, kernel_name in MODEL_OF_SMOOTH.items():
        kernel_cls, _, _ = get_kernel(kernel_name)
        int8 = kernel_cls()
        int8_torch_ppl[smooth] = pseudo_perplexity(lambda e: int8(e.float()), mlm_head, samples)
        print(f"int8 torch ({kernel_name}): {int8_torch_ppl[smooth]:.4f}", flush=True)

    torch.save({
        "n_eval": args.n_eval, "seed": args.seed, "calib_windows": args.calib_windows, "calib_seed": args.calib_seed,
        "samples": samples,
        "mlm_head": mlm_head.state_dict(), "mlm_head_dim": mlm_head.mlm_dense.in_features,
        "mlm_head_vocab": mlm_head.mlm_decoder.out_features,
        "ptf_alpha": calibrate_ptf_alpha(model, tok, hf_model, args.calib_windows, args.calib_seed),
        "float_ppl": float_ppl, "int8_torch_ppl": int8_torch_ppl,
    }, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
