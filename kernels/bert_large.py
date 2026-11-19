import math
import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig
from torch.ao.quantization.observer import default_per_channel_weight_observer
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from kernels.w3 import to_w3


# BERT-Large dimensions (matches bert-large-uncased: hidden=1024, 24 layers,
# 16 heads, intermediate=4096)
DIM = 1024
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
D_MODEL = DIM
NUM_HEADS = 16
HEAD_DIM = DIM // NUM_HEADS
FFN_DIM = 4 * DIM
NUM_LAYERS = 24
LAYERNORM_EPS = 1e-12


class BertSelfAttention(nn.Module):
    """Real multi-head attention (NUM_HEADS heads of HEAD_DIM, scaled by
    1/sqrt(HEAD_DIM)), split the same way as kernels/vit.py's ViTSelfAttention."""

    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(DIM, DIM, bias=True)
        self.Wk = nn.Linear(DIM, DIM, bias=True)
        self.Wv = nn.Linear(DIM, DIM, bias=True)
        self.Wo = nn.Linear(DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_lin = self.Wq(x)
        k_lin = self.Wk(x)
        v_lin = self.Wv(x)
        q_4d = q_lin.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        k_4d = k_lin.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        v_4d = v_lin.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM)
        q = q_4d.permute(0, 2, 1, 3)
        k_t = k_4d.permute(0, 2, 3, 1)
        v = v_4d.permute(0, 2, 1, 3)
        attn_scores = torch.matmul(q, k_t) * (1.0 / math.sqrt(HEAD_DIM))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 2, 1, 3).reshape(BATCH_SIZE, MAX_SEQ_LEN, DIM)
        return self.Wo(out)


class BertFFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, FFN_DIM, bias=True)
        self.fc2 = nn.Linear(FFN_DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class BertLargeBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = BertSelfAttention()
        self.ffn = BertFFN()
        # Real BERT is post-LN (add residual, then normalize), unlike
        # ViT/Llama's pre-LN -- and uses standard LayerNorm, not RMSNorm.
        self.ln1 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)
        self.ln2 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class BertLargeKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([BertLargeBlock() for _ in range(NUM_LAYERS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# The int8 bert-large: built once by run_scripts/measure_accuracy/build_bert_large_int8.py
# (real bert-large-uncased weights, calibrated on real WikiText-2 text) and loaded from
# here by the compiler.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "bert_large_int8.pt")
QUANTIZED_ENGINE = "qnnpack"


def quantize_bert_encoder(model: BertLargeKernel, calibration_inputs: list) -> torch.fx.GraphModule:
    """The quantization scheme, the same as kernels/vit.py's quantize_vit_encoder:
    every nn.Linear and torch.matmul to int8 by static post-training quantization
    (torch.ao quantize_fx, qnnpack: per-tensor affine activations; Linear weights
    symmetric with one scale per output channel; torch.matmul per-tensor).
    layer_norm, softmax, gelu and the residual adds stay float.
    calibration_inputs: encoder inputs [BATCH_SIZE, MAX_SEQ_LEN, DIM]."""
    torch.backends.quantized.engine = QUANTIZED_ENGINE
    per_tensor = get_default_qconfig(QUANTIZED_ENGINE)
    linear = QConfig(activation=per_tensor.activation, weight=default_per_channel_weight_observer)
    mapping = QConfigMapping().set_object_type(nn.Linear, linear).set_object_type(torch.matmul, per_tensor)
    prepared = prepare_fx(model.eval(), mapping, (calibration_inputs[0],))
    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)
    return convert_fx(prepared)


class BertLargeKernelInt8(nn.Module):
    """The saved int8 bert-large (INT8_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class BertLargeKernelW3(nn.Module):
    """The int8 bert-large (BertLargeKernelInt8) with group-wise 3-bit weights
    (kernels/w3.py): int8 activations, every Linear on 3-bit weights in groups of 64
    rows; the attention matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(BertLargeKernelInt8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
