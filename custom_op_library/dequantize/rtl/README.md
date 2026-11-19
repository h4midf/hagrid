# dequantize/rtl

Mirror of quantize/rtl, direction reversed: int8 `q` in, fixed-point `x` out (XW = 32:
Q12.20, XW = 16: Q8.8).
Six datapath variants, same matrix as quantize (trunc+zp combos skipped).

| File | Scale | Rounding | Zero point | Notes |
|---|---|---|---|---|
| dequant_mant_round.v | mantissa m, shift sh | round to nearest | no | parameter MW |
| dequant_mant_trunc.v | mantissa m, shift sh | truncate | no | parameter MW |
| dequant_pow2_round.v | shift sh | round to nearest | no | no multiplier |
| dequant_pow2_trunc.v | shift sh | truncate | no | shifter and clamp only |
| dequant_mant_round_zp.v | mantissa m, shift sh | round to nearest | yes | parameter MW |
| dequant_pow2_round_zp.v | shift sh | round to nearest | yes | no multiplier |

`m`/`sh` encode the dequantisation scale directly (`x = (q - zp) * m * 2^-sh`),
not quantize's `1/scale`: a mantissa + shift fit of the scale, computed per op by
compiler/ops/quant_encoding.py. Zero point is subtracted from `q` before the multiply, since it
lives in the int8 domain on the input side here (quantize adds it after
rounding, on the output side).

The mantissa variants can represent any positive scale magnitude (the free
mantissa `m` and shift `sh` jointly cover it). The pow2-only variants (no
mantissa) can only represent scales of the form `2^-sh`, i.e. scale <= 1 --
use a mantissa variant when the real dequantisation scale is larger.

Output is Q12.20 or Q8.8 (see ../../common/fixed_point.md), clamped to the full XW-bit
word range.

Shift range: `sh` must be 0..31, same reasoning as quantize/rtl/README.md.
