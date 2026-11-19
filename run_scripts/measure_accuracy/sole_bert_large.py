"""SOLE's E2Softmax and AILayerNorm (ICCAD 2023), and Wang et al.'s nonuniform-LUT GELU
(IEEE TCAD 2025, Algorithm 3; the journal version of SOLE, whose Algorithms 1-2 are
SOLE's), as real-valued torch functions for the accuracy sweep (variant_impl_reference.py).

E2Softmax (Algorithm 1, Eq. 7-13), with the row's global max in place of the online
running max (exact for a whole row):
  Log2Exp(x) = -floor(x + x/2 - x/16), x <= 0 (1/ln2 ~= 1.4375), clipped to 4 bits  (Eq. 8)
  ALDivision(k_y, S) = 2^-(k_y + k_s + 1) * (1.636 - q(s)), S = 2^k_s * (1 + s),
  q(s) = floor(2s) / 2 in {0, 0.5}; 1.636 is the paper's bias correction           (Eq. 13)

AILayerNorm (Algorithm 2, Eq. 6, 14-16): per-channel PTF int8 quantization (a power-of-two
exponent per channel, calibrated from the call's own input, may be negative), the exact
mean, and E(x^2) from DynamicCompress: an 8-bit magnitude >= 64 shifted right by 4, else
by 2, clipped to 4 bits, squared and shifted back (<< 4s), times 2^(2 alpha), and the
averaging step's << 4.

Wang GELU: x < -3 gives 0, x > 3 gives x, and between them a 24-bin table (edges, outputs).
"""
import torch


def log2exp(x: torch.Tensor) -> torch.Tensor:
    """Eq. 8 for x <= 0, clipped to the 4-bit range [0, 15]."""
    x = torch.clamp(x, max=0.0)
    k = -torch.floor(x + x / 2.0 - x / 16.0)
    return torch.clamp(k, 0.0, 15.0)


def aldivision(k_y: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Eq. 9-13: 2^-k_y / S through S = 2^k_s * (1 + s) and the bias-corrected 1-bit
    approximation of 1 / (1 + s)."""
    S = torch.clamp(S, min=1e-30)
    k_s = torch.floor(torch.log2(S))
    s = S / (2.0 ** k_s) - 1.0
    q = torch.floor(2.0 * s) / 2.0  # Eq. 10: q(s) in {0, 0.5}
    return (2.0 ** (-(k_y + k_s + 1.0))) * (1.636 - q)


def e2softmax(scores: torch.Tensor) -> torch.Tensor:
    """Algorithm 1 over the last dim."""
    shifted = scores - scores.max(dim=-1, keepdim=True).values
    k_y = log2exp(shifted)
    S = (2.0 ** (-k_y)).sum(dim=-1, keepdim=True)
    return aldivision(k_y, S)


def dynamic_compress(x_mag: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Eq. 15 on an 8-bit magnitude: (the 4-bit compressed value, the shift selector s)."""
    x_mag = torch.clamp(x_mag, 0.0, 255.0)
    high = x_mag >= 64.0  # x[7:6] != 0
    x_c = torch.where(high, torch.clamp(torch.floor(x_mag / 16.0), 0.0, 15.0),
                      torch.clamp(torch.floor(x_mag / 4.0), 0.0, 15.0))
    s = torch.where(high, torch.ones_like(x_mag), torch.zeros_like(x_mag))
    return x_c, s


def ptf_quantize(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """PTF (Eq. 6): one power-of-two exponent alpha per channel (the last dim), from the
    channel's max |x| over dims 0 and 1, and the symmetric int8 values. alpha may be
    negative: LayerNorm inputs are far below 127, so alpha >= 0 would round them to
    integers."""
    amax = x.abs().amax(dim=(0, 1), keepdim=True).clamp(min=1e-12)
    alpha = torch.ceil(torch.log2(amax / 127.0))
    x_q = torch.clamp(torch.round(x / 2.0 ** alpha), -128.0, 127.0)
    return x_q, alpha


def ailayernorm(x: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor, eps: float) -> torch.Tensor:
    """Algorithm 2. Stage 2 runs on the quantized, per-channel rescaled value x_q << alpha."""
    x_q, alpha = ptf_quantize(x)
    x_shifted = x_q * (2.0 ** alpha)
    ex = x_shifted.mean(dim=-1, keepdim=True)
    x_c, s = dynamic_compress(x_q.abs())
    x_c2 = (x_c ** 2) * (2.0 ** (4.0 * s))         # square and decompress
    x_c2_shifted = x_c2 * (2.0 ** (2.0 * alpha))   # << 2 alpha per element
    ex2 = x_c2_shifted.mean(dim=-1, keepdim=True) * 16.0   # the averaging step's << 4
    var = ex2 - ex ** 2
    std_inv = torch.rsqrt(torch.clamp(var, min=1e-12) + eps)
    return (x_shifted - ex) * std_inv * weight + bias


def wang_gelu(x: torch.Tensor, edges: torch.Tensor, outputs: torch.Tensor) -> torch.Tensor:
    """Algorithm 3: 0 below -3, x above 3, the table (25 edges, 24 outputs) between."""
    idx = torch.bucketize(torch.clamp(x, -3.0, 3.0), edges[1:-1])
    return torch.where(x < -3.0, torch.zeros_like(x), torch.where(x > 3.0, x, outputs[idx]))
