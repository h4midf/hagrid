// Dequantise, mantissa scale, with zero point: x = clamp(round((q - zp) * m * 2^-sh)).
// zp: signed 8-bit offset, subtracted from q BEFORE the multiply (zero point
// lives in the int8 domain on the input side here, unlike quantize where zp
// is added after rounding in the int8 domain on the output side).
module dequant_mant_round_zp #(parameter XW = 32, parameter MW = 12) (
    input  wire signed [7:0]  q,
    input  wire        [MW-1:0] m,
    input  wire        [5:0]  sh,
    input  wire signed [7:0]  zp,
    output wire signed [XW-1:0] x
);
    localparam signed [41:0] MAXV = (42'sd1 <<< (XW - 1)) - 1;
    localparam signed [41:0] MINV = -(42'sd1 <<< (XW - 1));

    wire signed [8:0]  qz   = $signed({q[7], q}) - $signed({zp[7], zp});
    wire signed [41:0] p    = qz * $signed({1'b0, m});
    wire signed [41:0] half = (sh == 0) ? 42'sd0 : (42'sd1 <<< (sh - 1));
    wire signed [41:0] r    = (p + half) >>> sh;
    assign x = (r > MAXV) ? MAXV[XW-1:0] : (r < MINV) ? MINV[XW-1:0] : r[XW-1:0];
endmodule
