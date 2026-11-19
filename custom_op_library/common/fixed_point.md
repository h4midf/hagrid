# Fixed-point format

Non-matmul values use Q12.20: 32-bit signed word, 12 integer bits including the sign, 20 fraction bits. Range -2048 to 2048 - 2^-20. The variants' other data width is Q8.8: 16-bit signed word, 8 fraction bits.

- Rounding: round to nearest, ties up (add half the discarded LSB before the shift).
- Saturation: clamp to the word range on every operation that can overflow.
- Int8 outputs (quantise): clamp to -128..127.
