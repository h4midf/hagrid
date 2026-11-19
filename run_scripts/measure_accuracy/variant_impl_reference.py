"""The paper circuits' gelu / softmax / layer_norm functions for the accuracy sweep
(evaluate_variant_configs.py), imported from each paper's
run_scripts/measure_accuracy/<paper>_bert_large.py so both compute the same thing.

QUARK's, SOLE's and I-BERT's are closed-form and re-exported as they are. GQA-LUT's and
Wang-TCAD's are calibrated tables: they use the frozen Q12.20 constants of their RTL
tests (custom_op_library/gelu/test_gqa_lut_gelu8.py, test_wang_tcad_gelu24.py,
custom_op_library/softmax/test_gqa_lut_exp8.py), because re-running GQA-LUT's genetic
search does not reproduce the synthesized breakpoints. Wang-TCAD's paper calibrates one
table per layer; the circuit (wang_tcad_gelu24.v) has one table for every layer, and so
does this function.
"""
import torch

from run_scripts.measure_accuracy.gqa_lut_bert_large import gqa_pwl, gqa_pwl_exp
from run_scripts.measure_accuracy.ibert_bert_large import i_gelu, i_softmax
from run_scripts.measure_accuracy.quark_bert_large import quark_gelu, quark_softmax
from run_scripts.measure_accuracy.sole_bert_large import ailayernorm, e2softmax, wang_gelu

FRAC = 20  # Q12.20 -- the W=32 characterized row every table below is transcribed from


def _q(ints: list[int]) -> torch.Tensor:
    return torch.tensor([v / float(1 << FRAC) for v in ints], dtype=torch.float64)


# custom_op_library/gelu/test_gqa_lut_gelu8.py's TABLES[32], as gqa_lut_gelu8.v hardwires
# it: breakpoints, slopes and intercepts all in Q12.20 (Eq. 3's quantization applied)
_GQA_GELU_BP = _q([-2883584, -2293760, -1146880, -786432, -262144, 491520, 2359296])
_GQA_GELU_SLOPE = _q([-5807, -42184, -117730, -65924, 143396, 610830, 1144232, 1062282])
_GQA_GELU_ICEPT = _q([-32768, -131072, -294912, -229376, -98304, 0, -262144, -65536])

# custom_op_library/softmax/test_gqa_lut_exp8.py's TABLES[32]
_GQA_EXP_BP = _q([-6160384, -3866624, -2686976, -1769472, -1048576, -589824, -262144])
_GQA_EXP_SLOPE = _q([1128, 9839, 47971, 127934, 275262, 480592, 699413, 922184])
_GQA_EXP_ICEPT = _q([0, 65536, 196608, 393216, 655360, 851968, 983040, 1048576])
_GQA_EXP_RN = -8.0

# custom_op_library/gelu/test_wang_tcad_gelu24.py's TABLES[32]
_WANG_BP_RAW = [-2916352, -2654208, -2424832, -2195456, -1933312, -1703936, -1474560, -1212416, -950272,
                -688128, -393216, -131072, 131072, 425984, 688128, 950272, 1245184, 1507328, 1769472,
                2064384, 2326528, 2588672, 2883584]
_WANG_OUT_RAW = [0, 0, -32768, -32768, -65536, -65536, -98304, -131072, -163840, -163840, -163840, -98304,
                 0, 163840, 393216, 655360, 950272, 1245184, 1540096, 1867776, 2162688, 2424832, 2719744,
                 3014656]
# wang_gelu takes 25 edges for 24 outputs: the 23 breakpoints between the +-3 domain
# fences (wang_tcad_gelu24.v's `dom`)
_WANG_EDGES = torch.cat([
    torch.tensor([-3.0], dtype=torch.float64), _q(_WANG_BP_RAW), torch.tensor([3.0], dtype=torch.float64),
])
_WANG_OUTPUTS = _q(_WANG_OUT_RAW)


def gqa_gelu(x: torch.Tensor) -> torch.Tensor:
    return gqa_pwl(x, _GQA_GELU_BP, _GQA_GELU_SLOPE, _GQA_GELU_ICEPT)


def gqa_softmax(scores: torch.Tensor) -> torch.Tensor:
    """Max-subtract, GQA-LUT exp (0 below the domain floor), divide by the sum."""
    x_max = scores.max(dim=-1, keepdim=True).values
    x_tilde = scores - x_max
    e = gqa_pwl_exp(x_tilde, _GQA_EXP_BP, _GQA_EXP_SLOPE, _GQA_EXP_ICEPT, r_n=_GQA_EXP_RN)
    return e / e.sum(dim=-1, keepdim=True)


def wang_tcad_gelu(x: torch.Tensor) -> torch.Tensor:
    return wang_gelu(x, _WANG_EDGES, _WANG_OUTPUTS)


# QUARK's, SOLE's and I-BERT's functions are re-exported as they are
__all__ = [
    "gqa_gelu", "gqa_softmax", "wang_tcad_gelu",
    "quark_gelu", "quark_softmax",
    "e2softmax", "ailayernorm",
    "i_gelu", "i_softmax",
]
