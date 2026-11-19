"""QUARK's shared base-2 log / exp circuit (ICCAD 2025) for softmax and GELU, as
real-valued torch functions for the accuracy sweep (variant_impl_reference.py).

Appro-Exp(x), x <= 0 (Eq. 3-5): x_shift = x + x/2 - x/16 (x * log2(e)), q_I = ceil(x_shift),
  x_F = x_shift - q_I, result (0.1713 x_F^2 + 0.6674 x_F + 0.998) * 2^q_I.
Appro-Ln(x), x > 0 (Eq. 6-9): x = 2^q_M * q_N, q_N in [1, 2),
  log2(x) = q_M - 0.3369 q_N^2 + 1.995 q_N - 1.65, ln(x) = log2(x) - log2(x)/4 - log2(x)/16.
Softmax (Eq. 2): x' = x - max(x), output Appro-Exp(x' - Appro-Ln(sum Appro-Exp(x'))).
GELU (Eq. 11-14): x * Softmax([0, -1.702x])[0] (the sigmoid as a 2-element softmax) for
  |x| < 2.4, ReLU(x) otherwise.
"""
import torch


def appro_exp(x: torch.Tensor) -> torch.Tensor:
    """Eq. 3-5 for x <= 0."""
    x = torch.clamp(x, max=0.0)
    x_shift = x + x / 2.0 - x / 16.0
    q_I = torch.ceil(x_shift)
    x_F = x_shift - q_I
    return (0.1713 * x_F ** 2 + 0.6674 * x_F + 0.998) * (2.0 ** q_I)


def appro_ln(x: torch.Tensor) -> torch.Tensor:
    """Eq. 6-9 for x > 0."""
    x = torch.clamp(x, min=1e-30)
    q_M = torch.floor(torch.log2(x))
    q_N = x / (2.0 ** q_M)
    log2_x = q_M + (-0.3369 * q_N ** 2 + 1.995 * q_N - 1.65)
    return log2_x - log2_x / 4.0 - log2_x / 16.0


def quark_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Eq. 2 over the last dim."""
    x_prime = scores - scores.max(dim=-1, keepdim=True).values
    S = appro_exp(x_prime).sum(dim=-1, keepdim=True)
    return appro_exp(x_prime - appro_ln(S))


def quark_gelu(x: torch.Tensor) -> torch.Tensor:
    """Eq. 11-14. The 1.702 scales only the sigmoid's input (x * sigma(1.702x))."""
    sigmoid_approx = quark_softmax(torch.stack([torch.zeros_like(x), -1.702 * x], dim=-1))[..., 0]
    return torch.where(x.abs() < 2.4, x * sigmoid_approx, torch.relu(x))
