// Quantise, full scale: q = clamp_int8(round(x * m * 2^-sh)).
// x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16). m: MW-bit unsigned
// mantissa. sh: right shift (range in README.md). Round to nearest, ties up.
module quant_mant_round #(parameter XW = 32, parameter MW = 12) (
    input  wire signed [XW-1:0] x,
    input  wire        [MW-1:0] m,
    input  wire        [5:0]  sh,
    output wire signed [7:0]  q
);
    localparam PW = 1 + XW + MW;
    wire signed [PW-1:0] p    = x * $signed({1'b0, m});
    wire signed [PW-1:0] half = (sh == 0) ? {PW{1'b0}} : ({{(PW-1){1'b0}}, 1'b1} << (sh - 1));
    wire signed [PW-1:0] r    = (p + half) >>> sh;
    assign q = (r > $signed(PW'sd127)) ? 8'sd127 : (r < $signed(-PW'sd128)) ? -8'sd128 : r[7:0];
endmodule
