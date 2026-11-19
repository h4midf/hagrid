"""GPT-2 Small (openai-community/gpt2): 12 decoder blocks, hidden 768, 12 heads of 64,
MLP 3072, context 1024. The kernel is the decoder stack plus the final LayerNorm
(ln_f) over one full 1024-token window (a prefill pass): the token and position
embeddings and the LM head (tied to the token embedding) run outside it, as BERT's
embeddings and MLM head do (kernels/bert_large.py).

Faithful to Hugging Face's GPT2Model:
  pre-LN blocks          x = x + attn(ln_1(x)); x = x + mlp(ln_2(x)); then ln_f
  causal self-attention  scores scaled by 1/sqrt(64), then the future masked as HF's
                         eager attention does: torch.where(causal_mask, scores,
                         finfo(float32).min), then softmax
  MLP                    c_fc, GELU in its tanh form (gelu_new), c_proj
  Wq / Wk / Wv           the three column blocks of HF's fused c_attn (a Conv1D, so its
                         weight is the transpose of nn.Linear's)
"""
import math
import os

import copy

import torch
import torch.fx
import torch.nn as nn

from kernels.w3 import GROUP_ROWS, WEIGHT_BITS, group_codes

DIM = 768
MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
NUM_HEADS = 12
HEAD_DIM = DIM // NUM_HEADS
FFN_DIM = 4 * DIM
NUM_LAYERS = 12
LAYERNORM_EPS = 1e-5


class GPT2Attention(nn.Module):
    """Causal multi-head self-attention, split like kernels/bert_large.py's."""

    def __init__(self):
        super().__init__()
        self.Wq = nn.Linear(DIM, DIM, bias=True)
        self.Wk = nn.Linear(DIM, DIM, bias=True)
        self.Wv = nn.Linear(DIM, DIM, bias=True)
        self.Wo = nn.Linear(DIM, DIM, bias=True)
        # True on and below the diagonal: the positions a token may attend to
        mask = torch.tril(torch.ones(MAX_SEQ_LEN, MAX_SEQ_LEN, dtype=torch.bool))
        self.register_buffer("causal_mask", mask)
        self.register_buffer("mask_value", torch.tensor(torch.finfo(torch.float32).min))

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
        attn_scores = torch.where(self.causal_mask, attn_scores, self.mask_value)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        out = torch.matmul(attn_probs, v)
        out = out.permute(0, 2, 1, 3).reshape(BATCH_SIZE, MAX_SEQ_LEN, DIM)
        return self.Wo(out)


class GPT2MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(DIM, FFN_DIM, bias=True)
        self.fc2 = nn.Linear(FFN_DIM, DIM, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)
        self.attn = GPT2Attention()
        self.ln2 = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)
        self.ffn = GPT2MLP()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2Kernel(nn.Module):
    """The decoder stack and ln_f; input: embedded tokens [BATCH_SIZE, MAX_SEQ_LEN, DIM]."""

    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([GPT2Block() for _ in range(NUM_LAYERS)])
        self.ln_f = nn.LayerNorm(DIM, eps=LAYERNORM_EPS)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)


# ---- int8: SmoothQuant W8A8 --------------------------------------------------------
# Every matmul is int8 x int8 with an int32 accumulator, dequantized to float (no
# requantization to int8):
#   Linear   activation per token (dynamic), weight per output channel
#   Q.K^T    Q per token, K per token
#   P.V      P per token, V per tensor (static, calibrated)
# LayerNorm, softmax, GELU, the causal mask and the residual adds stay float. A scale
# is carried as a maximum: a value with maximum M is quantized with scale M / 127.
# Each step is one wrapped function, so tracing keeps it one graph node, as torch.ao's
# quantize_per_tensor / quantized ops are for kernels/bert_large.py and kernels/vit.py
# (torch.ao has no per-token int8 Linear or matmul).
#
# The int8 GPT-2 is built once by run_scripts/measure_accuracy/build_gpt2_int8.py (real
# openai-community/gpt2 weights, calibrated on real WikiText-2 text) and loaded from
# INT8_CHECKPOINT by the compiler.
INT8_CHECKPOINT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "dump", "models",
                               "gpt2_int8.pt")
INT8_MAX = 127
SMOOTHQUANT_ALPHA = 0.5


def row_absmax(x: torch.Tensor) -> torch.Tensor:
    """max |x| over the last dim, one per row (custom_op_library/token_quant's
    absmax_reduce_tree)."""
    return x.abs().amax(dim=-1, keepdim=True)


def quantize_per_token(x: torch.Tensor, row_max: torch.Tensor) -> torch.Tensor:
    """int8 of x with one scale per row, row_max / 127 (token_quant's token_scale, then
    quantize's quant_mant_round)."""
    scale = row_max.clamp(min=torch.finfo(torch.float32).tiny) / INT8_MAX
    return torch.round(x / scale).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def quantize_per_tensor_static(x: torch.Tensor, x_max: torch.Tensor) -> torch.Tensor:
    """int8 of x with one calibrated scale, x_max / 127."""
    return torch.round(x / (x_max / INT8_MAX)).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def int8_linear(q: torch.Tensor, w_int8: torch.Tensor) -> torch.Tensor:
    """q @ w_int8^T as an int32 accumulator (the int8 x int8 products, exact in float64)."""
    return torch.matmul(q.to(torch.float64), w_int8.to(torch.float64).transpose(-1, -2)).to(torch.int32)


def int8_matmul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """a @ b of two int8 activations as an int32 accumulator."""
    return torch.matmul(a.to(torch.float64), b.to(torch.float64)).to(torch.int32)


def dequantize_per_token(acc: torch.Tensor, row_max: torch.Tensor, col_max: torch.Tensor,
                         bias: torch.Tensor | None = None) -> torch.Tensor:
    """acc * (row_max / 127) * (col_max / 127), plus bias: row_max one per row of acc,
    col_max one per column (an output channel or a token) or a single value."""
    out = acc.to(torch.float32) * (row_max / INT8_MAX) * (col_max / INT8_MAX)
    return out if bias is None else out + bias


torch.fx.wrap("row_absmax")
torch.fx.wrap("quantize_per_token")
torch.fx.wrap("quantize_per_tensor_static")
torch.fx.wrap("int8_linear")
torch.fx.wrap("int8_matmul")
torch.fx.wrap("dequantize_per_token")


def _smoothed_groups(model: GPT2Kernel) -> list:
    """(LayerNorm, the Linears reading its output) of every block: the layers
    SmoothQuant smooths (the ones whose input scale can be folded into a LayerNorm)."""
    groups = []
    for block in model.layers:
        groups.append((block.ln1, [block.attn.Wq, block.attn.Wk, block.attn.Wv]))
        groups.append((block.ln2, [block.ffn.fc1]))
    return groups


def smooth_gpt2(model: GPT2Kernel, calibration_inputs: list, alpha: float = SMOOTHQUANT_ALPHA) -> None:
    """SmoothQuant (Xiao et al., ICML 2023), in place: for each LayerNorm and the
    Linears reading it, input channel j gets s_j = max|X_j|^alpha / max|W_j|^(1-alpha)
    (X_j: the LayerNorm output over the calibration inputs; W_j: column j of every
    Linear in the group). The LayerNorm's weight and bias are divided by s and the
    Linears' columns multiplied by s, so the float model is unchanged and the outlier
    channels move from the activations into the weights."""
    groups = _smoothed_groups(model)
    act_max = {id(ln): None for ln, _ in groups}

    def hook(ln):
        def record(_module, _inputs, output):
            m = output.detach().abs().reshape(-1, output.shape[-1]).amax(dim=0)
            act_max[id(ln)] = m if act_max[id(ln)] is None else torch.maximum(act_max[id(ln)], m)
        return record

    handles = [ln.register_forward_hook(hook(ln)) for ln, _ in groups]
    with torch.no_grad():
        for x in calibration_inputs:
            model(x)
    for h in handles:
        h.remove()

    with torch.no_grad():
        for ln, linears in groups:
            w_max = torch.stack([lin.weight.abs().amax(dim=0) for lin in linears]).amax(dim=0)
            s = (act_max[id(ln)].pow(alpha) / w_max.pow(1 - alpha)).clamp(min=1e-5)
            ln.weight.div_(s)
            ln.bias.div_(s)
            for lin in linears:
                lin.weight.mul_(s)


class Int8Linear(nn.Module):
    """A Linear with int8 weights (symmetric, one maximum per output channel) whose input
    is quantized per token."""

    def __init__(self, lin: nn.Linear):
        super().__init__()
        w = lin.weight.detach().float()
        w_max = w.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
        w_int8 = torch.round(w / (w_max[:, None] / INT8_MAX)).clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        self.register_buffer("w_int8", w_int8)
        self.register_buffer("w_max", w_max)
        self.register_buffer("bias", lin.bias.detach().float().clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = row_absmax(x)
        acc = int8_linear(quantize_per_token(x, x_max), self.w_int8)
        return dequantize_per_token(acc, x_max, self.w_max, self.bias)


class GPT2AttentionInt8(nn.Module):
    """GPT2Attention with every matmul int8 (see the int8 section above)."""

    def __init__(self, attn: GPT2Attention, v_max: float):
        super().__init__()
        self.Wq, self.Wk, self.Wv, self.Wo = (Int8Linear(attn.Wq), Int8Linear(attn.Wk),
                                              Int8Linear(attn.Wv), Int8Linear(attn.Wo))
        self.register_buffer("causal_mask", attn.causal_mask.clone())
        self.register_buffer("mask_value", attn.mask_value.clone())
        self.register_buffer("v_max", torch.tensor(float(v_max)))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        q = self.Wq(x).view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        k = self.Wk(x).view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        v = self.Wv(x).view(BATCH_SIZE, MAX_SEQ_LEN, NUM_HEADS, HEAD_DIM).permute(0, 2, 1, 3)
        # Q.K^T: Q and K per token (K's token scales are the output's column scales)
        q_max = row_absmax(q)
        k_max = row_absmax(k)
        acc = int8_matmul(quantize_per_token(q, q_max), quantize_per_token(k, k_max).transpose(-1, -2))
        attn_scores = dequantize_per_token(acc, q_max, k_max.transpose(-1, -2)) * (1.0 / math.sqrt(HEAD_DIM))
        attn_scores = torch.where(self.causal_mask, attn_scores, self.mask_value)
        attn_probs = torch.softmax(attn_scores, dim=-1)
        # P.V: P per token, V per tensor (a per-token V scale would sit inside the sum)
        p_max = row_absmax(attn_probs)
        acc = int8_matmul(quantize_per_token(attn_probs, p_max), quantize_per_tensor_static(v, self.v_max))
        out = dequantize_per_token(acc, p_max, self.v_max)
        out = out.permute(0, 2, 1, 3).reshape(BATCH_SIZE, MAX_SEQ_LEN, DIM)
        return self.Wo(out)


class GPT2MLPInt8(nn.Module):
    def __init__(self, ffn: GPT2MLP):
        super().__init__()
        self.fc1 = Int8Linear(ffn.fc1)
        self.fc2 = Int8Linear(ffn.fc2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x, approximate="tanh")
        x = self.fc2(x)
        return x


class GPT2BlockInt8(nn.Module):
    def __init__(self, block: GPT2Block, v_max: float):
        super().__init__()
        self.ln1 = copy.deepcopy(block.ln1)
        self.attn = GPT2AttentionInt8(block.attn, v_max)
        self.ln2 = copy.deepcopy(block.ln2)
        self.ffn = GPT2MLPInt8(block.ffn)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class GPT2Int8(nn.Module):
    """The int8 decoder stack and ln_f, from a smoothed GPT2Kernel and each layer's
    calibrated max |V|."""

    def __init__(self, model: GPT2Kernel, v_max: list):
        super().__init__()
        self.layers = nn.ModuleList([GPT2BlockInt8(b, vm) for b, vm in zip(model.layers, v_max)])
        self.ln_f = copy.deepcopy(model.ln_f)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x)
        return self.ln_f(x)


def calibrate_v_max(model: GPT2Kernel, calibration_inputs: list) -> list:
    """Each layer's max |V| (the Wv output) over the calibration inputs: V's static scale."""
    v_max = [0.0] * len(model.layers)
    handles = [block.attn.Wv.register_forward_hook(
        lambda _m, _i, out, li=li: v_max.__setitem__(li, max(v_max[li], float(out.abs().max()))))
        for li, block in enumerate(model.layers)]
    with torch.no_grad():
        for x in calibration_inputs:
            model(x)
    for h in handles:
        h.remove()
    return v_max


def quantize_gpt2_decoder(model: GPT2Kernel, calibration_inputs: list) -> GPT2Int8:
    """SmoothQuant W8A8 (the int8 section above): smooth_gpt2 (in place), V's static
    maxima, then the int8 model. calibration_inputs: embedded windows
    [BATCH_SIZE, MAX_SEQ_LEN, DIM]."""
    smooth_gpt2(model, calibration_inputs)
    return GPT2Int8(model, calibrate_v_max(model, calibration_inputs)).eval()


class GPT2KernelInt8(nn.Module):
    """The saved int8 GPT-2 (INT8_CHECKPOINT); raises if it has not been built."""

    def __init__(self):
        super().__init__()
        self._quantized = torch.load(INT8_CHECKPOINT, weights_only=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)


# Group-wise 3-bit weights (kernels/w3.py's format) on the per-token int8 path: each
# Linear's weight is split into one maximum per output channel (w_max, as Int8Linear's)
# and, per GROUP_ROWS input rows, a factor group_scale * 127 / w_max, so the groups'
# rescaled sum is in Int8Linear's accumulator units and dequantize_per_token is unchanged.
def w3_linear_raw(q: torch.Tensor, codes: torch.Tensor, group_factor: torch.Tensor, weight_bits: int,
                  group_rows: int) -> torch.Tensor:
    """q (int8, per token) @ codes^T, each group of group_rows rows' accumulator times its
    factor (group_factor [out, groups]), summed over the groups."""
    k = codes.shape[1]
    w = codes.to(torch.float64) * group_factor.to(torch.float64).repeat_interleave(group_rows, dim=1)[:, :k]
    return torch.matmul(q.to(torch.float64), w.transpose(-1, -2)).to(torch.float32)


torch.fx.wrap("w3_linear_raw")


class W3PerTokenLinear(nn.Module):
    """Int8Linear with group-wise 3-bit weights: per-token int8 input, w3_linear_raw, then
    dequantize_per_token with the per-channel maxima."""

    def __init__(self, lin: Int8Linear):
        super().__init__()
        w = lin.w_int8.float() * (lin.w_max[:, None] / INT8_MAX)
        w_max = w.abs().amax(dim=1).clamp(min=torch.finfo(torch.float32).tiny)
        codes, group_scale = group_codes(w)
        self.register_buffer("codes", codes)
        self.register_buffer("group_factor", group_scale * INT8_MAX / w_max[:, None])
        self.register_buffer("w_max", w_max)
        self.register_buffer("bias", lin.bias.clone())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_max = row_absmax(x)
        acc = w3_linear_raw(quantize_per_token(x, x_max), self.codes, self.group_factor, WEIGHT_BITS, GROUP_ROWS)
        return dequantize_per_token(acc, x_max, self.w_max, self.bias)


class GPT2KernelW3(nn.Module):
    """The int8 GPT-2 (GPT2KernelInt8) with every Int8Linear on group-wise 3-bit weights
    (W3PerTokenLinear); the attention matmuls stay int8."""

    def __init__(self):
        super().__init__()
        self._quantized = GPT2KernelInt8()._quantized
        for name, sub in list(self._quantized.named_modules()):
            if type(sub) is Int8Linear:
                parent_name, _, child = name.rpartition(".")
                setattr(self._quantized.get_submodule(parent_name), child, W3PerTokenLinear(sub))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._quantized(x)
