"""BERT-Base's attention alone (no FFN, no LayerNorm): the int8 12-layer attention app
(the paper's mapping motivation, Fig. 3), its first layer alone (Fig. 6), and its w3
form."""
import copy
import os

import torch
import torch.nn as nn

from kernels.bert_base import (
    BATCH_SIZE, DIM, MAX_SEQ_LEN, SMOOTHQUANT_ALPHA, BertMultiHeadAttention, _smoothing_scale,
)
from kernels.bert_large import QUANTIZED_ENGINE, quantize_bert_encoder
from kernels.w3 import to_w3

NUM_LAYERS = 12


# ---- the int8 12-layer attention app (the paper's mapping motivation, Fig. 3)
#
# bert-base's real attention (BertMultiHeadAttention, NUM_HEADS heads) of each of its 12
# layers, stacked with no FFN, LayerNorm or residual between them, quantized with
# bert-base's scheme (kernels/bert_base.py): SmoothQuant at SMOOTHQUANT_ALPHA on each
# layer's Q/K/V input, then quantize_bert_encoder's static int8 of every Linear and matmul.
# It is calibrated through this stack, not cut out of bert-base's int8 checkpoint: there,
# each layer's input scales are calibrated on LayerNorm outputs, while here a layer's
# input is the previous layer's attention output.


class AttentionStack(nn.Module):
    """Attention layers applied one after another."""

    def __init__(self, layers: list):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


class SmoothedAttention(nn.Module):
    """An attention layer with SmoothQuant's 1 / s before Wq / Wk / Wv (their columns
    multiplied by s); the float function is unchanged."""

    def __init__(self, attn: BertMultiHeadAttention, s_qkv: torch.Tensor):
        super().__init__()
        self.attn = copy.deepcopy(attn)
        with torch.no_grad():
            for lin in (self.attn.Wq, self.attn.Wk, self.attn.Wv):
                lin.weight.mul_(s_qkv)
        self.register_buffer("inv_s_qkv", (1.0 / s_qkv).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.attn(x * self.inv_s_qkv)


def quantize_attention_stack(stack: AttentionStack, calibration_inputs: list, alpha: float) -> torch.fx.GraphModule:
    """The int8 attention stack from the float `stack` (float32, BertMultiHeadAttention
    layers): per-channel max |X| of each layer's input over calibration_inputs
    ([BATCH_SIZE, MAX_SEQ_LEN, DIM]), the smoothed stack, then quantize_bert_encoder."""
    act_max = {}

    def record(li):
        def hook(_module, inputs):
            m = inputs[0].detach().abs().reshape(-1, inputs[0].shape[-1]).amax(dim=0)
            act_max[li] = m if li not in act_max else torch.maximum(act_max[li], m)
        return hook

    handles = [layer.register_forward_pre_hook(record(li)) for li, layer in enumerate(stack.layers)]
    with torch.no_grad():
        for x in calibration_inputs:
            stack(x)
    for h in handles:
        h.remove()
    smoothed = [SmoothedAttention(layer, _smoothing_scale(act_max[li], [layer.Wq, layer.Wk, layer.Wv], alpha))
                for li, layer in enumerate(stack.layers)]
    return quantize_bert_encoder(AttentionStack(smoothed).eval(), calibration_inputs)


# built once by run_scripts/measure_accuracy/build_bert_attention_12layer_int8.py
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "bert_attention_12layer_int8.pt")
# the same app with only its first layer (the paper's mapping example, Fig. 6): built by the
# same script with --layers 1, so it equals the 12-layer app's first layer (both are
# calibrated on bert-base's embeddings)
INT8_1LAYER_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                                      "bert_attention_1layer_int8.pt")


class BertAttention12LayerKernelInt8(nn.Module):
    """The saved int8 12-layer attention app (INT8_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class BertAttention1LayerKernelInt8(nn.Module):
    """The saved int8 one-layer attention app (INT8_1LAYER_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_1LAYER_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class BertAttention12LayerKernelW3(nn.Module):
    """The int8 12-layer attention app with group-wise 3-bit weights (kernels/w3.py), for
    the RRAM/SRAM ensemble; the attention matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(BertAttention12LayerKernelInt8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
