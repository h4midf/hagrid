"""Torch models of the non-matmul impl choices and the shared bert-base evaluation helpers.

  softmax_impl / gelu_impl / layer_norm_impl  each impl on real values (the emulator,
                                              bert_base_int8_emulator.py, runs them on
                                              its fixed-point values); the paper circuits
                                              come from variant_impl_reference.py
  MLMHead, load_real_weights                  bert-base-uncased's weights (Hugging Face)
                                              on BertBaseKernel's names, and its MLM head
  windows, masked_inputs, real_embed          WikiText-2 windows, 15% masking, and BERT's
                                              embeddings (outside the traced kernel)
"""
import math

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fn
from huggingface_hub import hf_hub_download
from transformers import AutoModelForMaskedLM, AutoTokenizer

from kernels.bert_base import DIM, LAYERNORM_EPS, NUM_LAYERS
from variant_generators.common import LUT_TABLE_DEFAULTS

MASK_ID, CLS_ID, SEP_ID = 103, 101, 102
RSQRT_LUT_N = 1024


def softmax_impl(x: torch.Tensor, impl: str) -> torch.Tensor:
    if impl == "exact":
        return torch.softmax(x, dim=-1)
    if impl == "lut":
        # exp table over [xmin, 0] and reciprocal table over [1, T], nearest index, with
        # the compiler's table settings (LUT_TABLE_DEFAULTS["softmax"])
        d = LUT_TABLE_DEFAULTS["softmax"]
        xmin, nbins, rcp_nbins = d["xmin"], d["nbins"], d["rcp_nbins"]
        s = x - x.max(dim=-1, keepdim=True).values
        tab = torch.exp(torch.linspace(xmin, 0.0, nbins, dtype=x.dtype))
        idx = torch.clamp(torch.round((s - xmin) * (nbins - 1) / (-xmin)).long(), 0, nbins - 1)
        w = tab[idx]
        total = w.sum(dim=-1, keepdim=True)
        t = x.shape[-1]
        rtab = 1.0 / torch.linspace(1.0, t, rcp_nbins, dtype=x.dtype)
        ridx = torch.clamp(torch.round((torch.clamp(total, 1.0, t) - 1.0) * (rcp_nbins - 1) / (t - 1.0)).long(),
                            0, rcp_nbins - 1)
        return w * rtab[ridx]
    # the paper circuits. Function-local: variant_impl_reference.py and the <paper>_bert_large
    # scripts import masked_inputs / real_embed / windows from this module, so a top-level
    # import would be circular
    from run_scripts.measure_accuracy.variant_impl_reference import e2softmax, gqa_softmax
    from run_scripts.measure_accuracy.quark_bert_large import quark_softmax
    from run_scripts.measure_accuracy.ibert_bert_large import i_softmax
    if impl == "sole":
        return e2softmax(x)
    if impl == "quark":
        return quark_softmax(x)
    if impl == "gqa_lut":
        return gqa_softmax(x)
    if impl == "ibert":
        return i_softmax(x)
    raise ValueError(impl)


def gelu_impl(x: torch.Tensor, impl: str) -> torch.Tensor:
    if impl == "exact":
        return Fn.gelu(x)
    if impl == "lut":
        # the compiler's table settings (LUT_TABLE_DEFAULTS["gelu"])
        d = LUT_TABLE_DEFAULTS["gelu"]
        xmin, xmax, nbins = d["xmin"], d["xmax"], d["nbins"]
        tab = Fn.gelu(torch.linspace(xmin, xmax, nbins, dtype=x.dtype))
        idx = torch.clamp(torch.round((x - xmin) * (nbins - 1) / (xmax - xmin)).long(), 0, nbins - 1)
        return tab[idx]
    # the paper circuits, function-local as in softmax_impl
    from run_scripts.measure_accuracy.variant_impl_reference import gqa_gelu, wang_tcad_gelu
    from run_scripts.measure_accuracy.quark_bert_large import quark_gelu
    from run_scripts.measure_accuracy.ibert_bert_large import i_gelu
    if impl == "quark":
        return quark_gelu(x)
    if impl == "wang_tcad":
        return wang_tcad_gelu(x)
    if impl == "gqa_lut":
        return gqa_gelu(x)
    if impl == "ibert":
        return i_gelu(x)
    raise ValueError(impl)


def rsqrt_impl(var: torch.Tensor, impl: str) -> torch.Tensor:
    """1/sqrt(var) for "exact", "rsqrt" and "lut". layer_norm_impl handles "sole"
    itself, since its normalization does not factor through a bare rsqrt(var)."""
    if impl == "exact":
        return torch.rsqrt(var)
    if impl == "rsqrt":
        # the CMOS rsqrt circuit (custom_op_library/rsqrt, ../layer_norm/rtl/rsqrt_q12_20.v):
        # var = 2^p * m, m in [1, 2), p = 2e + parity; a seed 1/sqrt(2^parity * m_5bit)
        # from a 32-entry table per parity, shifted by e, then one Newton iteration
        p = torch.floor(torch.log2(var))
        m = var / torch.pow(2.0, p)
        idx = torch.clamp(torch.floor((m - 1.0) * 32.0), 0, 31)
        parity = torch.remainder(p, 2.0)
        e = torch.floor(p / 2.0)
        y = torch.pow(2.0, -e) / torch.sqrt(torch.pow(2.0, parity) * (1.0 + idx / 32.0))
        return y * (1.5 - 0.5 * var * y * y)
    if impl == "lut":
        k = torch.floor(torch.log(var) / math.log(4.0))
        m = var / torch.pow(4.0, k)
        tab = 1.0 / torch.sqrt(1.0 + 3.0 * torch.arange(RSQRT_LUT_N, dtype=var.dtype) / (RSQRT_LUT_N - 1))
        idx = torch.clamp(torch.round((m - 1.0) * (RSQRT_LUT_N - 1) / 3.0).long(), 0, RSQRT_LUT_N - 1)
        return tab[idx] * torch.pow(2.0, -k)
    raise ValueError(impl)


def layer_norm_impl(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float, impl: str) -> torch.Tensor:
    # "sole" restructures the whole mean/center/scale chain. Function-local import: a
    # top-level one would be circular (see softmax_impl).
    if impl == "sole":
        from run_scripts.measure_accuracy.variant_impl_reference import ailayernorm
        return ailayernorm(x, weight, bias, eps)
    mu = x.mean(dim=-1, keepdim=True)
    xc = x - mu
    var = (xc * xc).mean(dim=-1, keepdim=True) + eps
    return xc * rsqrt_impl(var, impl) * weight + bias


class MLMHead(nn.Module):
    """BERT's masked-LM head (dense -> gelu -> LayerNorm -> decoder over the vocabulary),
    run outside the traced encoder: the hardware does not run it."""

    def __init__(self, dim: int, vocab_size: int):
        super().__init__()
        self.mlm_dense = nn.Linear(dim, dim)
        # BERT's layer_norm_eps (1e-12), not nn.LayerNorm's default 1e-5
        self.mlm_layer_norm = nn.LayerNorm(dim, eps=LAYERNORM_EPS)
        self.mlm_decoder = nn.Linear(dim, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = Fn.gelu(self.mlm_dense(x))
        h = self.mlm_layer_norm(h)
        return self.mlm_decoder(h)


def load_real_weights(kernel: nn.Module) -> tuple:
    """bert-base-uncased's Hugging Face weights loaded onto BertBaseKernel's parameter
    names, and its MLM head. Returns (tokenizer, the Hugging Face model, the MLM head)."""
    tok = AutoTokenizer.from_pretrained("bert-base-uncased")
    # double precision: in float32, 12 layers and a 30522-wide vocabulary miss the 1e-3
    # logit check against Hugging Face
    hf =AutoModelForMaskedLM.from_pretrained("bert-base-uncased").eval().double()
    sd = hf.state_dict()

    def g(name):
        return sd[name].detach().clone()

    sdict = kernel.state_dict()
    for l in range(NUM_LAYERS):
        p = f"bert.encoder.layer.{l}."
        sdict[f"layers.{l}.attn.Wq.weight"] = g(p + "attention.self.query.weight")
        sdict[f"layers.{l}.attn.Wq.bias"] = g(p + "attention.self.query.bias")
        sdict[f"layers.{l}.attn.Wk.weight"] = g(p + "attention.self.key.weight")
        sdict[f"layers.{l}.attn.Wk.bias"] = g(p + "attention.self.key.bias")
        sdict[f"layers.{l}.attn.Wv.weight"] = g(p + "attention.self.value.weight")
        sdict[f"layers.{l}.attn.Wv.bias"] = g(p + "attention.self.value.bias")
        sdict[f"layers.{l}.attn.Wo.weight"] = g(p + "attention.output.dense.weight")
        sdict[f"layers.{l}.attn.Wo.bias"] = g(p + "attention.output.dense.bias")
        sdict[f"layers.{l}.ln1.weight"] = g(p + "attention.output.LayerNorm.weight")
        sdict[f"layers.{l}.ln1.bias"] = g(p + "attention.output.LayerNorm.bias")
        sdict[f"layers.{l}.ffn.fc1.weight"] = g(p + "intermediate.dense.weight")
        sdict[f"layers.{l}.ffn.fc1.bias"] = g(p + "intermediate.dense.bias")
        sdict[f"layers.{l}.ffn.fc2.weight"] = g(p + "output.dense.weight")
        sdict[f"layers.{l}.ffn.fc2.bias"] = g(p + "output.dense.bias")
        sdict[f"layers.{l}.ln2.weight"] = g(p + "output.LayerNorm.weight")
        sdict[f"layers.{l}.ln2.bias"] = g(p + "output.LayerNorm.bias")
    kernel.load_state_dict(sdict)

    vocab_size = int(sd["cls.predictions.decoder.weight"].shape[0])
    mlm_head = MLMHead(DIM, vocab_size).eval().double()
    hsdict = mlm_head.state_dict()
    hsdict["mlm_dense.weight"] = g("cls.predictions.transform.dense.weight")
    hsdict["mlm_dense.bias"] = g("cls.predictions.transform.dense.bias")
    hsdict["mlm_layer_norm.weight"] = g("cls.predictions.transform.LayerNorm.weight")
    hsdict["mlm_layer_norm.bias"] = g("cls.predictions.transform.LayerNorm.bias")
    hsdict["mlm_decoder.weight"] = g("cls.predictions.decoder.weight")
    hsdict["mlm_decoder.bias"] = g("cls.predictions.bias")  # the decoder's bias is tied to this one
    mlm_head.load_state_dict(hsdict)
    return tok, hf, mlm_head


def windows(split, tok, n, L=126, seed=0):
    """n shuffled WikiText-2 windows of L tokens, each as [CLS] + tokens + [SEP]."""
    p = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    text = " ".join(t for t in pd.read_parquet(p)["text"] if t.strip())
    ids = tok(text, add_special_tokens=False, verbose=False)["input_ids"]
    w = [[CLS_ID] + ids[i:i + L] + [SEP_ID] for i in range(0, len(ids) - L, L)]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(w))[:n]
    return [w[i] for i in idx]


def masked_inputs(win, rng):
    """15% of a window's tokens (not [CLS] / [SEP], at least one) masked: (ids, positions)."""
    T = len(win)
    cand = np.arange(1, T - 1)
    k = max(1, int(round(0.15 * len(cand))))
    pos = np.sort(rng.choice(cand, size=k, replace=False))
    ids = np.array(win)
    ids[pos] = MASK_ID
    return ids, pos


def real_embed(hf_model, ids: np.ndarray) -> torch.Tensor:
    """BERT's embeddings (word + position + token type, then LayerNorm), which the
    kernel does not include."""
    emb = hf_model.bert.embeddings
    ids_t = torch.tensor(ids, dtype=torch.long).unsqueeze(0)
    pos_ids = torch.arange(len(ids), dtype=torch.long).unsqueeze(0)
    type_ids = torch.zeros_like(ids_t)
    with torch.no_grad():
        e = emb.word_embeddings(ids_t) + emb.position_embeddings(pos_ids) + emb.token_type_embeddings(type_ids)
        e = emb.LayerNorm(e)
    return e
