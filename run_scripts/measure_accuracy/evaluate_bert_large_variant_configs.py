"""bert-large-uncased's Hugging Face weights for build_bert_large_int8.py: the counterpart
of evaluate_variant_configs.load_real_weights, with bert-large's checkpoint and layer
count (the parameter names and the MLM head are the same as bert-base's).
"""
import torch.nn as nn
from transformers import AutoModelForMaskedLM, AutoTokenizer

from kernels.bert_large import DIM, NUM_LAYERS
from run_scripts.measure_accuracy.evaluate_variant_configs import MLMHead


def load_real_weights(kernel: nn.Module) -> tuple:
    """bert-large-uncased's weights loaded onto `kernel`'s parameter names
    (BertLargeKernel's), and its MLM head. Returns (tokenizer, the Hugging Face model,
    the MLM head)."""
    tok = AutoTokenizer.from_pretrained("bert-large-uncased")
    hf = AutoModelForMaskedLM.from_pretrained("bert-large-uncased").eval().double()
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
