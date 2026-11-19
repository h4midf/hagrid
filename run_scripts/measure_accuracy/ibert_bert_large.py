"""I-BERT's integer-only GELU and softmax (ICML 2021), as real-valued torch functions for
the accuracy sweep (variant_impl_reference.py).

GELU (Eq. 8-9): x * 0.5 * (1 + L(x / sqrt(2))), L(u) = sgn(u) * (a (min(|u|, -b) + b)^2 + c),
  a = -0.2888, b = -1.769, c = 1.
Softmax (Eq. 10-14): x~ = x - max(x) = -ln2 * z + p, z = floor(-x~ / ln2), p in (-ln2, 0];
  exp(p) ~= 0.3585 (p + 1.353)^2 + 0.344, exp(x~) = exp(p) * 2^-z, then divide by the sum.
"""
import math

import torch

LN2 = math.log(2.0)


def i_gelu(x: torch.Tensor) -> torch.Tensor:
    """Eq. 8-9."""
    a, b, c = -0.2888, -1.769, 1.0
    u = x / math.sqrt(2.0)
    mag = torch.clamp(u.abs(), max=-b)
    return x * 0.5 * (1.0 + torch.sign(u) * (a * (mag + b) ** 2 + c))


def i_exp(x_tilde: torch.Tensor) -> torch.Tensor:
    """Eq. 12-13 for x_tilde <= 0."""
    x_tilde = torch.clamp(x_tilde, max=0.0)
    z = torch.floor(-x_tilde / LN2)
    p = x_tilde + z * LN2
    return (0.3585 * (p + 1.353) ** 2 + 0.344) * (2.0 ** (-z))


def i_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Eq. 10-14 over the last dim."""
    e = i_exp(scores - scores.max(dim=-1, keepdim=True).values)
    return e / e.sum(dim=-1, keepdim=True)
