// Dequantise, mantissa scale, truncation: x = clamp(floor((q - zp) * m * 2^-sh)), zp = 0 here.
// No rounding adder. Biased toward -inf by up to one LSB of the output.
// See dequant_mant_round.v for the port/width conventions (m/sh encode the
// dequantisation scale directly; internal width is a fixed 42 bits).
module dequant_mant_trunc #(parameter XW = 32, parameter MW = 12) (
    input  wire signed [7:0]  q,
    input  wire        [MW-1:0] m,
    input  wire        [5:0]  sh,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz = {q[7], q};
    wire signed [41:0] p  = qz * $signed({1'b0, m});
    wire signed [41:0] r  = p >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
