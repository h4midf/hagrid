// Division of a row sum by the row length ROW, as the streamed layer norms use it:
// q = (sum * RECIP) >>> RS with RECIP = round(2^RS / ROW), the low QW bits (floor; for
// a power-of-two ROW it equals sum >>> log2(ROW)). One constant multiplier.
module row_div #(parameter AW = 43, parameter QW = 32, parameter ROW = 768, parameter RS = 40) (
    input  wire signed [AW-1:0] sum,
    output wire signed [QW-1:0] q
);
    localparam [63:0] RECIP = ((64'd1 << RS) + ROW / 2) / ROW;
    wire signed [AW+RS+1:0] prod = $signed(sum) * $signed({1'b0, RECIP[RS:0]});
    wire signed [AW+RS+1:0] quot = prod >>> RS;
    assign q = quot[QW-1:0];
endmodule
