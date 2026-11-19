# vec_select/rtl

| File | Computes | Notes |
|---|---|---|
| vec_select.v | out[i] = mask[i] ? x[i] : fill | one 2-to-1 multiplexer per lane; mask one bit per lane; fill one W-bit word shared by the lanes (runtime input); parameters N, W |

`torch.where(cond, x, c)` with a scalar `c`. In GPT-2 it is the causal mask: `cond` is the stored lower-triangular mask (one bit per score), `x` the attention scores, `fill` the most negative word of the format (Q12.20 at W = 32: -2048; the model's float32 minimum is not representable), which softmax turns into a zero weight, exactly as the float model's -3.4e38 does.

Input and output are W-bit signed (Q12.20 at W = 32, Q8.8 at W = 16, see ../../common/fixed_point.md); a select does not change the format.

Test: `../test_vec_select.py` (Yosys eval against a Python model, bit-exact).
