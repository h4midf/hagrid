# quantize/rtl

Six datapath variants. Per-channel and per-row quantisation use the same modules: the caller supplies the mantissa `m` and shift `sh` for the current channel or row from a scale table, which is memory and is not in this library.

| File | Scale | Rounding | Zero point | Notes |
|---|---|---|---|---|
| quant_mant_round.v | mantissa m, shift sh | round to nearest | no | parameter MW |
| quant_mant_trunc.v | mantissa m, shift sh | truncate | no | parameter MW |
| quant_pow2_round.v | shift sh | round to nearest | no | no multiplier |
| quant_pow2_trunc.v | shift sh | truncate | no | shifter and clamp only |
| quant_mant_round_zp.v | mantissa m, shift sh | round to nearest | yes | parameter MW |
| quant_pow2_round_zp.v | shift sh | round to nearest | yes | no multiplier |

Output is int8 with saturation. Input is `XW`-bit signed (Q12.20 at XW=32, Q8.8 at XW=16, see ../../common/fixed_point.md).

Shift range: `sh` is 0..min(31, container-2), where `container` is the module's intermediate width (`XW` for the pow2 variants, `1+XW+MW` for the mantissa ones), so the rounding term `2^(sh-1)` fits as a positive value. At XW=32 the bound is 31; at XW=16 it is narrower.
