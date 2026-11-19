// Quantise, power-of-two scale, truncation: q = clamp_int8(floor(x * 2^-sh)).
// Only an arithmetic shifter and the clamp.
// x: XW-bit signed (Q12.20 at XW=32, Q8.8 at XW=16).
module quant_pow2_trunc #(parameter XW = 32) (
    input  wire signed [XW-1:0] x,
    input  wire        [5:0]  sh,
    output wire signed [7:0]  q
);
    wire signed [XW-1:0] r = x >>> sh;
    assign q = (r > $signed(XW'sd127)) ? 8'sd127 : (r < $signed(-XW'sd128)) ? -8'sd128 : r[7:0];
endmodule
