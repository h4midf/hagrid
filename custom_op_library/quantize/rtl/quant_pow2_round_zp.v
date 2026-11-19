// Quantise, power-of-two scale, with zero point: q = clamp_int8(round(x * 2^-sh) + zp).
// x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16).
module quant_pow2_round_zp #(parameter XW = 32) (
    input  wire signed [XW-1:0] x,
    input  wire        [5:0]  sh,
    input  wire signed [7:0]  zp,
    output wire signed [7:0]  q
);
    localparam PW = XW + 1;
    wire signed [XW-1:0] half = (sh == 0) ? {XW{1'b0}} : ({{(XW-1){1'b0}}, 1'b1} <<< (sh - 1));
    wire signed [PW-1:0] sum  = $signed({x[XW-1], x}) + $signed({half[XW-1], half});
    wire signed [PW-1:0] r0   = sum >>> sh;
    wire signed [PW-1:0] r    = r0 + $signed({{(PW-8){zp[7]}}, zp});
    assign q = (r > $signed(PW'sd127)) ? 8'sd127 : (r < $signed(-PW'sd128)) ? -8'sd128 : r[7:0];
endmodule
