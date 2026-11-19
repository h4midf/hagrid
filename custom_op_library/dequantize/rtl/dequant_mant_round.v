// Dequantise, mantissa scale: x = clamp(round((q - zp) * m * 2^-sh)), zp = 0 here.
// q: int8 signed. m: MW-bit unsigned mantissa. sh: right shift, 0..31.
// Output x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16), clamped to
// the full word range.
//
// m/sh encode the dequantisation scale directly (x = q * scale), not quantize's
// 1/scale: a mantissa + shift fit of the scale, computed per op by
// compiler/ops/quant_encoding.py. The free mantissa covers any positive scale
// magnitude (not just scale <= 1), unlike the pow2 variants.
//
// Internal width is a fixed 42 bits regardless of MW or XW (covers the
// worst case, MW=31: 9 (q-zp) + 32 (m, zero-extended) = 41, +1 guard bit;
// also comfortably exceeds XW+1 at either characterized XW), so the final
// clamp can always safely slice the low XW bits of the shifted result.
module dequant_mant_round #(parameter XW = 32, parameter MW = 12) (
    input  wire signed [7:0]  q,
    input  wire        [MW-1:0] m,
    input  wire        [5:0]  sh,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz   = {q[7], q};
    wire signed [41:0] p    = qz * $signed({1'b0, m});
    wire signed [41:0] half = (sh == 0) ? 42'sd0 : (42'sd1 <<< (sh - 1));
    wire signed [41:0] r    = (p + half) >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
