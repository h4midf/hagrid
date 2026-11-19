import math
import os

import torch
import torch.nn as nn

from kernels.vit import QUANTIZED_ENGINE
from kernels.w3 import to_w3


# ViT-Base encoder dimensions aligned with google/vit-base-patch16-224
DIM = 768
MAX_SEQ_LEN = 197
BATCH_SIZE = 1
D_MODEL = DIM
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS
MLP_DIM = 4 * DIM
NUM_LAYERS = 12


class ViTSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(DIM, DIM, bias=True)
        self.Wk = nn.Linear(DIM, DIM, bias=True)
        self.Wv = nn.Linear(DIM, DIM, bias=True)
        self.proj = nn.Linear(DIM, DIM, bias=True)

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
        # The second residual connects to o3, as in HF's ViTOutput.
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


# The int8 vit-base: built once by run_scripts/measure_accuracy/build_vit_base_int8.py
# (real google/vit-base-patch16-224 weights, calibrated on real ImageNet images).
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "vit_base_int8.pt")


class ViTBaseEncoderInt8(nn.Module):
    """The saved int8 vit-base (INT8_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class ViTBaseEncoderW3(nn.Module):
    """The int8 vit-base (ViTBaseEncoderInt8) with group-wise 3-bit weights
    (kernels/w3.py): int8 activations, every Linear on 3-bit weights in groups of 64
    rows; the attention matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(ViTBaseEncoderInt8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
