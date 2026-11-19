"""GQA-LUT's piecewise-linear evaluation (DAC 2024, Eq. 1) for the accuracy sweep
(variant_impl_reference.py): pwl_i(x) = k_i * x + b_i on [p_(i-1), p_i). The tables
(breakpoints, slopes, intercepts) are the frozen ones of the RTL tests.
"""
import torch


def gqa_pwl(x: torch.Tensor, breakpoints: torch.Tensor, slopes: torch.Tensor, intercepts: torch.Tensor) -> torch.Tensor:
    """x is not clamped: past the outer breakpoints the first / last segment extrapolates
    (right for GELU, which is ~x for large x)."""
    idx = torch.searchsorted(breakpoints, x, right=True)
    return slopes[idx] * x + intercepts[idx]


def gqa_pwl_exp(x: torch.Tensor, breakpoints: torch.Tensor, slopes: torch.Tensor, intercepts: torch.Tensor,
                r_n: float) -> torch.Tensor:
    """gqa_pwl, with 0 below the domain floor r_n: exp saturates to 0, and extrapolating
    the first segment there gave bert-large a perplexity of 5.7e4."""
    out = gqa_pwl(x, breakpoints, slopes, intercepts)
    return torch.where(x < r_n, torch.zeros_like(out), out)
