// Dequantise, power-of-two scale, with zero point: x = clamp(round((q - zp) * 2^-sh)).
// zp subtracted from q before the shift (see dequant_mant_round_zp.v's header
// comment for why this is on the input side here, unlike quantize). Output
// x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16).
module dequant_pow2_round_zp #(parameter XW = 32) (
    input  wire signed [7:0]  q,
    input  wire        [5:0]  sh,
    input  wire signed [7:0]  zp,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz   = $signed({q[7], q}) - $signed({zp[7], zp});
    wire signed [41:0] half = (sh == 0) ? 42'sd0 : (42'sd1 <<< (sh - 1));
    wire signed [41:0] sum  = $signed({{33{qz[8]}}, qz}) + half;
    wire signed [41:0] r    = sum >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
