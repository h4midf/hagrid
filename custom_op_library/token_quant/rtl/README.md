# token_quant/rtl

Per-token (dynamic, one scale per row) int8 quantisation of an activation row, the activation side of W8A8 with per-channel weights (SmoothQuant's default). Two new units; the per-element rounding is `../../quantize/rtl/quant_mant_round.v`.

```
x row (W-bit signed) ─► absmax_reduce_tree ─► M ─► token_scale ─► (m, sh) ─► quant_mant_round per element ─► int8 row
```

| File | Computes | Per | Notes |
|---|---|---|---|
| absmax_reduce_tree.v | M = max \|x_i\| over N lanes | pass of N lanes | binary tree as `softmax/rtl/max_reduce_tree.v`, \|x\| at the leaves, unsigned output (\|-2^(W-1)\| exact); parameters N (power of two), W |
| token_scale.v | m, sh with m * 2^-sh ~= 127 / M | row | sh = min(bitlen(M) + MW - 8, SH_MAX), m = round(127 * 2^sh / M); m in [2032, 4064] at MW = 12; M = 0 gives m = sh = 0; parameters W, MW, SH_MAX (31, quant_mant_round's limit) |

x and M share one fixed-point format (Q12.20 at W = 32, see ../../common/fixed_point.md), so the factor 127 / M depends only on the integer M.

On the output side no row scale needs computing: the int32 accumulator of the int8 matmul is multiplied by M itself and by the per-channel constant w_scale / 127 (folded offline).

Tests (Yosys eval against `../token_quant_ref.py`, bit-exact):
- `../test_absmax_reduce_tree.py`: N = 8 and 256, W = 32.
- `../test_token_scale.py`: every bit length of M and edge cases at MW = 12; m < 2^MW; and the whole path (absmax, token_scale, quant_mant_round) within one int8 step of round(x * 127 / M).
