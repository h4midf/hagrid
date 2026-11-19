import math
import os

import torch
import torch.nn as nn
from torch.ao.quantization import QConfig, QConfigMapping, get_default_qconfig
from torch.ao.quantization.observer import default_per_channel_weight_observer
from torch.ao.quantization.quantize_fx import convert_fx, prepare_fx

from kernels.w3 import to_w3


# ViT-Large encoder dimensions aligned with google/vit-large-patch16-224
# (confirmed directly against AutoConfig.from_pretrained): 196 patches
# (14x14 at patch16 on a 224px image) + 1 CLS token = 197, same patch16/
# 224 setup kernels/vit_base.py's own MAX_SEQ_LEN=197 already uses.
DIM = 1024
MAX_SEQ_LEN = 197
BATCH_SIZE = 1
D_MODEL = DIM
NUM_HEADS = 16
HEAD_DIM = DIM // NUM_HEADS
MLP_DIM = 4 * DIM
NUM_LAYERS = 24


class ViTSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(DIM, DIM, bias=True)
        self.Wk = nn.Linear(DIM, DIM, bias=True)
        self.Wv = nn.Linear(DIM, DIM, bias=True)
        self.proj = nn.Linear(DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q_lin = self.Wq(x)
        # return q_lin
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
        return self.proj(out)


class ViTMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, MLP_DIM, bias=True)
        self.fc2 = nn.Linear(MLP_DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x


class ViTBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-12)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-12)
        self.attn = ViTSelfAttention()
        self.mlp = ViTMLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.norm1(x)
        o2 = self.attn(o1)
        o3 = x + o2
        o4 = self.norm2(o3)
        o5 = self.mlp(o4)
        # the second residual adds the first residual's output o3, not the block input
        # x, as Hugging Face's ViTOutput(hidden_states, input_tensor)
        o6 = o3 + o5
        return o6


class ViTEncoder(nn.Module):
    def __init__(self, num_layers: int = NUM_LAYERS):
        super().__init__()
        self.layers = nn.ModuleList([ViTBlock() for _ in range(num_layers)])
        self.final_norm = nn.LayerNorm(DIM, eps=1e-12)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.final_norm(x)


# The int8 vit-large: built once by run_scripts/measure_accuracy/build_vit_large_int8.py
# (real google/vit-large-patch16-224 weights, calibrated on real ImageNet images) and
# loaded from here by both the compiler and the accuracy evaluation.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "vit_large_int8.pt")
QUANTIZED_ENGINE = "qnnpack"


def quantize_vit_encoder(model: ViTEncoder, calibration_inputs: list) -> torch.fx.GraphModule:
    """The quantization scheme (its one definition): every nn.Linear and torch.matmul
    to int8 by static post-training quantization (torch.ao quantize_fx, qnnpack:
    per-tensor affine activations; Linear weights symmetric with one scale per output
    channel; torch.matmul has no weights, its two activations per-tensor). The trace
    then has quantize_per_tensor -> quantized Linear / quantized.matmul (int8 x int8 ->
    i32 inside) -> dequantize; layer_norm, softmax, gelu and the residual adds stay
    float. calibration_inputs: encoder inputs [BATCH_SIZE, MAX_SEQ_LEN, DIM] the
    observers see."""
    torch.backends.quantized.engine = QUANTIZED_ENGINE
    per_tensor = get_default_qconfig(QUANTIZED_ENGINE)
    linear = QConfig(activation=per_tensor.activation, weight=default_per_channel_weight_observer)
    mapping = QConfigMapping().set_object_type(nn.Linear, linear).set_object_type(torch.matmul, per_tensor)
    prepared = prepare_fx(model.eval(), mapping, (calibration_inputs[0],))
    with torch.no_grad():
        for x in calibration_inputs:
            prepared(x)
    return convert_fx(prepared)


class ViTEncoderInt8(nn.Module):
    """The saved int8 vit-large (INT8_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class ViTEncoderW3(nn.Module):
    """The int8 vit-large (ViTEncoderInt8) with group-wise 3-bit weights (kernels/w3.py):
    int8 activations, every Linear on 3-bit weights in groups of 64 rows; the attention
    matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(ViTEncoderInt8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)

