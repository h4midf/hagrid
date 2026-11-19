"""Builds the int8 GPT-2 that the compiler uses: real openai-community/gpt2 weights,
quantized with kernels/gpt2.py's scheme (quantize_gpt2_decoder: SmoothQuant W8A8),
calibrated on real WikiText-2 text, saved to kernels.gpt2.INT8_CHECKPOINT.

Windows are MAX_SEQ_LEN (1024) consecutive GPT-2 tokens of WikiText-2 (raw); the token
and position embeddings and the LM head (tied to the token embedding) run outside the
kernel. Calibration uses the train split, the perplexity the test split. Before
quantizing, the float kernel + LM head is checked against Hugging Face's logits.
Reports WikiText-2 perplexity of the float and the int8 model (next-token prediction
within each window, as HyFlexPIM evaluates GPT-2: WikiText-2, MSL 1024).

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_gpt2_int8 --n-calib 16 --calib-seed 1 --n-eval 16
"""
import argparse
import math
import os

import numpy as np
import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer, GPT2LMHeadModel

from kernels.gpt2 import INT8_CHECKPOINT, MAX_SEQ_LEN, NUM_LAYERS, GPT2Kernel, quantize_gpt2_decoder

HF_NAME = "openai-community/gpt2"
MAX_LOGIT_DIFF = 1e-3   # float kernel vs Hugging Face, in double


def load_real_weights(kernel: GPT2Kernel) -> tuple:
    """Maps the HF GPT-2 weights onto the kernel. HF's attention and MLP projections are
    Conv1D (weight [in, out]); nn.Linear's weight is [out, in], so each is transposed,
    and c_attn's [768, 2304] is split into Wq, Wk, Wv by columns."""
    tok = AutoTokenizer.from_pretrained(HF_NAME)
    hf = GPT2LMHeadModel.from_pretrained(HF_NAME).eval().double()
    sd = hf.state_dict()

    def g(name):
        return sd[name].detach().clone()

    dim = kernel.ln_f.normalized_shape[0]
    sdict = kernel.state_dict()
    for l in range(NUM_LAYERS):
        p = f"transformer.h.{l}."
        w, b = g(p + "attn.c_attn.weight"), g(p + "attn.c_attn.bias")
        for i, name in enumerate(("Wq", "Wk", "Wv")):
            sdict[f"layers.{l}.attn.{name}.weight"] = w[:, i * dim:(i + 1) * dim].t().contiguous()
            sdict[f"layers.{l}.attn.{name}.bias"] = b[i * dim:(i + 1) * dim].clone()
        sdict[f"layers.{l}.attn.Wo.weight"] = g(p + "attn.c_proj.weight").t().contiguous()
        sdict[f"layers.{l}.attn.Wo.bias"] = g(p + "attn.c_proj.bias")
        sdict[f"layers.{l}.ffn.fc1.weight"] = g(p + "mlp.c_fc.weight").t().contiguous()
        sdict[f"layers.{l}.ffn.fc1.bias"] = g(p + "mlp.c_fc.bias")
        sdict[f"layers.{l}.ffn.fc2.weight"] = g(p + "mlp.c_proj.weight").t().contiguous()
        sdict[f"layers.{l}.ffn.fc2.bias"] = g(p + "mlp.c_proj.bias")
        for ln in ("ln1", "ln2"):
            hf_ln = {"ln1": "ln_1", "ln2": "ln_2"}[ln]
            sdict[f"layers.{l}.{ln}.weight"] = g(p + hf_ln + ".weight")
            sdict[f"layers.{l}.{ln}.bias"] = g(p + hf_ln + ".bias")
    sdict["ln_f.weight"] = g("transformer.ln_f.weight")
    sdict["ln_f.bias"] = g("transformer.ln_f.bias")
    kernel.load_state_dict(sdict)
    return tok, hf


def windows(split: str, tok, n: int, seed: int) -> list:
    """n random windows of MAX_SEQ_LEN consecutive tokens of the WikiText-2 (raw) split."""
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet",
                        repo_type="dataset")
    text = "".join(pd.read_parquet(p)["text"])
    ids = tok(text, verbose=False)["input_ids"]
    w = [ids[i:i + MAX_SEQ_LEN] for i in range(0, len(ids) - MAX_SEQ_LEN, MAX_SEQ_LEN)]
    idx = np.random.default_rng(seed).permutation(len(w))[:n]
    return [w[i] for i in idx]


def embed(hf, win: list) -> torch.Tensor:
    """Token + position embeddings (outside the kernel)."""
    ids = torch.tensor([win])
    with torch.no_grad():
        return hf.transformer.wte(ids) + hf.transformer.wpe(torch.arange(len(win)).unsqueeze(0))


def lm_logits(hf, hidden: torch.Tensor) -> torch.Tensor:
    """The LM head (tied to the token embedding)."""
    return hidden.double() @ hf.transformer.wte.weight.double().t()


def perplexity(decoder, hf, wins: list, dtype) -> float:
    nll = count = 0
    with torch.no_grad():
        for win in wins:
            logits = lm_logits(hf, decoder(embed(hf, win).to(dtype)))[0]
            target = torch.tensor(win[1:])
            nll += float(torch.nn.functional.cross_entropy(logits[:-1], target, reduction="sum"))
            count += len(target)
    return math.exp(nll / count)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="WikiText-2 train windows the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration windows")
    parser.add_argument("--n-eval", type=int, required=True, help="WikiText-2 test windows for the perplexity")
    args = parser.parse_args()

    model = GPT2Kernel().eval().double()
    tok, hf = load_real_weights(model)

    test = windows("test", tok, args.n_eval, seed=99)
    with torch.no_grad():
        mine = lm_logits(hf, model(embed(hf, test[0])))[0]
        ref = hf(input_ids=torch.tensor([test[0]])).logits[0]
    max_diff = float((mine - ref).abs().max())
    print(f"check vs HF logits ({len(test[0])} tokens): max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"GPT2Kernel + LM head does not match Hugging Face gpt2 (diff {max_diff})")

    model = model.float()
    hf = hf.float()
    calib = [embed(hf, w) for w in windows("train", tok, args.n_calib, seed=args.calib_seed)]
    fp = perplexity(model, hf, test, torch.float32)
    # quantize_gpt2_decoder smooths the model in place (SmoothQuant); the float model
    # must be unchanged by it
    quantized = quantize_gpt2_decoder(model, calib)
    sp = perplexity(model, hf, test, torch.float32)
    qp = perplexity(quantized, hf, test, torch.float32)
    print(f"WikiText-2 test perplexity over {len(test)} windows of {MAX_SEQ_LEN} tokens: "
          f"float {fp:.2f}, float after smoothing {sp:.2f}, int8 (SmoothQuant W8A8) {qp:.2f}")

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized, INT8_CHECKPOINT)
    print(f"wrote {INT8_CHECKPOINT}")


if __name__ == "__main__":
    main()
