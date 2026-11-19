// Dequantise, power-of-two scale: x = clamp(round((q - zp) * 2^-sh)), zp = 0 here.
// No multiplier. Only represents real scales of the form 2^-sh (scale <= 1) --
// the mantissa variant is needed when the real dequantisation scale is larger.
// Output x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16). Internal width
// is a fixed 42 bits regardless of XW (qz is only 9 bits, so this is cheap
// and sidesteps any sh-vs-width rounding-term overflow concern entirely).
module dequant_pow2_round #(parameter XW = 32) (
    input  wire signed [7:0]  q,
    input  wire        [5:0]  sh,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz   = {q[7], q};
    wire signed [41:0] half = (sh == 0) ? 42'sd0 : (42'sd1 <<< (sh - 1));
    wire signed [41:0] sum  = $signed({{33{qz[8]}}, qz}) + half;
    wire signed [41:0] r    = sum >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
