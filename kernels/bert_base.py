import copy
import math
import os

import torch
import torch.fx
import torch.nn as nn

from kernels.bert_large import QUANTIZED_ENGINE, quantize_bert_encoder
from kernels.gpt2 import INT8_MAX, dequantize_per_token, int8_linear, quantize_per_tensor_static
from kernels.w3 import to_w3

torch.fx.wrap("quantize_per_tensor_static")
torch.fx.wrap("int8_linear")
torch.fx.wrap("dequantize_per_token")


# BERT-Base dimensions (matches bert-base-uncased: hidden=768, 12 layers,
# 12 heads, intermediate=3072)
DIM = 768
MAX_SEQ_LEN = 512
BATCH_SIZE = 1
D_MODEL = DIM
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS
FFN_DIM = 4 * DIM
NUM_LAYERS = 12
LAYERNORM_EPS = 1e-12


class BertSelfAttention(nn.Module):
    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(DIM, DIM, bias=True)
        self.Wk = nn.Linear(DIM, DIM, bias=True)
        self.Wv = nn.Linear(DIM, DIM, bias=True)
        self.Wo = nn.Linear(DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x)
        k = self.Wk(x)
        v = self.Wv(x)
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * (1.0 / math.sqrt(DIM))
        attn_probs = torch.softmax(attn_scores, dim=-1)
        return self.Wo(torch.matmul(attn_probs, v))


class BertMultiHeadAttention(nn.Module):
    """Real multi-head attention (NUM_HEADS heads of HEAD_DIM, scaled by
    1/sqrt(HEAD_DIM)). BertSelfAttention above stays combined-head for
    kernels/bert_attention_only.py."""

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


class BertBaseBlock(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn = BertMultiHeadAttention()
        self.ffn = BertFFN()
        # Post-LN (add residual, then normalize), matching real BERT.
        self.ln1 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)
        self.ln2 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        attn_out = self.attn(x)
        x = self.ln1(x + attn_out)
        ffn_out = self.ffn(x)
        x = self.ln2(x + ffn_out)
        return x


class BertBaseKernel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([BertBaseBlock() for _ in range(NUM_LAYERS)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


# ---- the int8 bert-base: SmoothQuant W8A8, every matmul int8
#
# SmoothQuant (Xiao et al., ICML 2023) at SMOOTHQUANT_ALPHA on the inputs of Wq / Wk / Wv
# (the block input), fc1 (ln1's output) and fc2 (gelu's output): input channel j gets
# s_j = max|X_j|^alpha / max|W_:,j|^(1-alpha) over the calibration inputs, the Linears'
# columns are multiplied by s, and the input is multiplied by 1 / s explicitly (a
# per-channel mul before the quantize). In post-LN BERT, the LayerNorm output is also the
# residual, so 1 / s cannot be folded into the LayerNorm. The float model is unchanged.
#
# fc2's output is not requantized: it feeds only the residual add, so its int32
# accumulator is rescaled straight to real units (Int8RawLinear). Every other Linear and
# matmul is torch.ao's static int8 (quantize_bert_encoder).
SMOOTHQUANT_ALPHA = 0.75


class Int8RawLinear(nn.Module):
    """A Linear on int8 with a real-valued output: the input quantized per tensor with
    the calibrated maximum x_max (symmetric, scale x_max / 127), int8 weights (symmetric,
    one maximum per output channel), the int32 accumulator rescaled to real units plus
    bias (kernels/gpt2.py's int8 functions)."""

    def __init__(self, weight: torch.Tensor, bias: torch.Tensor, x_max: float):
        super().__init__()
        w = weight.detach().float()
        w_max = w.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
        self.register_buffer("w_int8", torch.round(w / (w_max[:, None] / INT8_MAX)).clamp(-INT8_MAX, INT8_MAX)
                             .to(torch.int8))
        self.register_buffer("w_max", w_max)
        self.register_buffer("bias", bias.detach().float().clone())
        self.register_buffer("x_max", torch.tensor(float(x_max)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        acc = int8_linear(quantize_per_tensor_static(x, self.x_max), self.w_int8)
        return dequantize_per_token(acc, self.x_max, self.w_max, self.bias)


class BertBaseBlockSmoothed(nn.Module):
    """BertBaseBlock with SmoothQuant's 1 / s before Wq / Wk / Wv, fc1 and fc2 (their
    columns multiplied by s) and fc2 an Int8RawLinear."""

    def __init__(self, block: BertBaseBlock, s_qkv: torch.Tensor, s_fc1: torch.Tensor, s_fc2: torch.Tensor,
                 fc2_x_max: float):
        super().__init__()
        self.attn = copy.deepcopy(block.attn)
        self.ln1 = copy.deepcopy(block.ln1)
        self.ln2 = copy.deepcopy(block.ln2)
        self.fc1 = copy.deepcopy(block.ffn.fc1)
        with torch.no_grad():
            for lin in (self.attn.Wq, self.attn.Wk, self.attn.Wv):
                lin.weight.mul_(s_qkv)
            self.fc1.weight.mul_(s_fc1)
        self.fc2 = Int8RawLinear(block.ffn.fc2.weight * s_fc2, block.ffn.fc2.bias, fc2_x_max)
        self.register_buffer("inv_s_qkv", (1.0 / s_qkv).float())
        self.register_buffer("inv_s_fc1", (1.0 / s_fc1).float())
        self.register_buffer("inv_s_fc2", (1.0 / s_fc2).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x * self.inv_s_qkv))
        h = torch.nn.functional.gelu(self.fc1(x * self.inv_s_fc1))
        return self.ln2(x + self.fc2(h * self.inv_s_fc2))


class BertBaseBlockFfnSmoothed(nn.Module):
    """BertBaseBlockSmoothed without the Wq / Wk / Wv smoothing: 1 / s only before fc1
    and fc2 (Wq / Wk / Wv read the block input as it is)."""

    def __init__(self, block: BertBaseBlock, s_fc1: torch.Tensor, s_fc2: torch.Tensor, fc2_x_max: float):
        super().__init__()
        self.attn = copy.deepcopy(block.attn)
        self.ln1 = copy.deepcopy(block.ln1)
        self.ln2 = copy.deepcopy(block.ln2)
        self.fc1 = copy.deepcopy(block.ffn.fc1)
        with torch.no_grad():
            self.fc1.weight.mul_(s_fc1)
        self.fc2 = Int8RawLinear(block.ffn.fc2.weight * s_fc2, block.ffn.fc2.bias, fc2_x_max)
        self.register_buffer("inv_s_fc1", (1.0 / s_fc1).float())
        self.register_buffer("inv_s_fc2", (1.0 / s_fc2).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.ln1(x + self.attn(x))
        h = torch.nn.functional.gelu(self.fc1(x * self.inv_s_fc1))
        return self.ln2(x + self.fc2(h * self.inv_s_fc2))


# the inputs SmoothQuant smooths: "qkv_fc1_fc2" (BertBaseBlockSmoothed) or "fc1_fc2"
# (BertBaseBlockFfnSmoothed: 12 fewer per-channel x / s ops)
SMOOTH_GROUPS = ("qkv_fc1_fc2", "fc1_fc2")


class BertBaseKernelSmoothed(nn.Module):
    def __init__(self, blocks: list):
        super().__init__()
        self.layers = nn.ModuleList(blocks)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return x


def _smoothing_scale(act_max: torch.Tensor, linears: list, alpha: float) -> torch.Tensor:
    w_max = torch.stack([lin.weight.detach().abs().amax(dim=0) for lin in linears]).amax(dim=0)
    return (act_max.pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)


def quantize_bert_base_encoder(model: BertBaseKernel, calibration_inputs: list,
                               alpha: float, smooth: str) -> torch.fx.GraphModule:
    """The int8 bert-base (the section above) from the float `model` (float32):
    per-channel max |X| of the smoothed inputs over calibration_inputs (encoder inputs
    [BATCH_SIZE, MAX_SEQ_LEN, DIM]), the smoothed model (`smooth`, one of SMOOTH_GROUPS),
    then quantize_bert_encoder's static int8 of every other Linear and matmul."""
    if smooth not in SMOOTH_GROUPS:
        raise ValueError(f"smooth={smooth!r} is not one of {SMOOTH_GROUPS}")
    act_max = {}

    def record(key):
        def hook(_module, inputs):
            m = inputs[0].detach().abs().reshape(-1, inputs[0].shape[-1]).amax(dim=0)
            act_max[key] = m if key not in act_max else torch.maximum(act_max[key], m)
        return hook

    handles = []
    for li, block in enumerate(model.layers):
        handles.append(block.attn.register_forward_pre_hook(record(("qkv", li))))
        handles.append(block.ffn.fc1.register_forward_pre_hook(record(("fc1", li))))
        handles.append(block.ffn.fc2.register_forward_pre_hook(record(("fc2", li))))
    with torch.no_grad():
        for x in calibration_inputs:
            model(x)
    for h in handles:
        h.remove()

    blocks = []
    for li, block in enumerate(model.layers):
        s_fc1 = _smoothing_scale(act_max[("fc1", li)], [block.ffn.fc1], alpha)
        s_fc2 = _smoothing_scale(act_max[("fc2", li)], [block.ffn.fc2], alpha)
        fc2_x_max = float((act_max[("fc2", li)] / s_fc2).max())
        if smooth == "qkv_fc1_fc2":
            s_qkv = _smoothing_scale(act_max[("qkv", li)], [block.attn.Wq, block.attn.Wk, block.attn.Wv], alpha)
            blocks.append(BertBaseBlockSmoothed(block, s_qkv, s_fc1, s_fc2, fc2_x_max))
        else:
            blocks.append(BertBaseBlockFfnSmoothed(block, s_fc1, s_fc2, fc2_x_max))
    return quantize_bert_encoder(BertBaseKernelSmoothed(blocks).eval(), calibration_inputs)


# The int8 bert-base: built once per SMOOTH_GROUPS entry by
# run_scripts/measure_accuracy/build_bert_base_int8.py (real bert-base-uncased weights,
# calibrated on real WikiText-2 text).
_MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models")
INT8_CHECKPOINTS = {
    "qkv_fc1_fc2": os.path.join(_MODELS_DIR, "bert_base_int8.pt"),
    "fc1_fc2": os.path.join(_MODELS_DIR, "bert_base_int8_ffn_smooth.pt"),
}


class BertBaseKernelInt8(nn.Module):
    """The saved int8 bert-base, SmoothQuant on Wq / Wk / Wv, fc1 and fc2
    (INT8_CHECKPOINTS["qkv_fc1_fc2"]); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINTS["qkv_fc1_fc2"], weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class BertBaseKernelInt8FfnSmooth(nn.Module):
    """The saved int8 bert-base, SmoothQuant on fc1 and fc2 only
    (INT8_CHECKPOINTS["fc1_fc2"]); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        torch.backends.quantized.engine = QUANTIZED_ENGINE
        self._quantized = torch.load(INT8_CHECKPOINTS["fc1_fc2"], weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


class BertBaseKernelW3(nn.Module):
    """The int8 bert-base (BertBaseKernelInt8) with group-wise 3-bit weights
    (kernels/w3.py): int8 activations, every Linear on 3-bit weights in groups of 64
    rows; the attention matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = to_w3(BertBaseKernelInt8()._quantized)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
