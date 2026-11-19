// Dequantise, power-of-two scale, truncation: x = clamp(floor((q - zp) * 2^-sh)), zp = 0 here.
// Only an arithmetic shifter and the clamp. Output x: XW-bit signed
// (Q12.20 at XW=32, Q8.8 at XW=16).
module dequant_pow2_trunc #(parameter XW = 32) (
    input  wire signed [7:0]  q,
    input  wire        [5:0]  sh,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz = {q[7], q};
    wire signed [41:0] r  = $signed({{33{qz[8]}}, qz}) >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
