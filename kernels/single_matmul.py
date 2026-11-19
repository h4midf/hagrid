import math

import torch
import torch.nn as nn


DIM = 3072
OUTPUT_DIM = 768
MAX_SEQ_LEN = 197
BATCH_SIZE = 1
NUM_HEADS = 12
HEAD_DIM = OUTPUT_DIM // NUM_HEADS
MLP_DIM = 4 * DIM
NUM_LAYERS = 12


class TransformerBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.norm1 = nn.LayerNorm(DIM, eps=1e-12)
        self.norm2 = nn.LayerNorm(DIM, eps=1e-12)
        self.Wq = nn.Linear(DIM, OUTPUT_DIM, bias=True)
        self.Wk = nn.Linear(DIM, OUTPUT_DIM, bias=True)
        self.Wv = nn.Linear(DIM, OUTPUT_DIM, bias=True)
        self.Wo = nn.Linear(OUTPUT_DIM, DIM, bias=True)
        self.fc1 = nn.Linear(DIM, MLP_DIM, bias=True)
        self.fc2 = nn.Linear(MLP_DIM, DIM, bias=True)

    def attention(self, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)

        q = q.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k_t = k.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 3, 1)
        v = v.view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)

        scores = torch.matmul(q, k_t) * (1.0 / math.sqrt(HEAD_DIM))
        probabilities = torch.softmax(scores, dim=-1)
        context = torch.matmul(probabilities, v)

        context = context.permute(0, 2, 1, 3).reshape(BATCH_SIZE, MAX_SEQ_LEN, OUTPUT_DIM)
        return self.Wo(context)

    def mlp(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.fc2(x)
        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        o1 = self.norm1(x)
        o2 = self.attention(o1)
        o3 = x + o2
        o4 = self.norm2(o3)
        o5 = self.mlp(o4)
        o6 = o3 + o5
        return o6


class AttentionKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([TransformerBlock() for _ in range(NUM_LAYERS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
