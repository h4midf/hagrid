// Q8.8 reciprocal of a positive Q8.8 sum, 65536/sum: softmax_row8.v's normalize
// stage with the sum width SUM_W as a parameter (a row of N terms sums to up to
// N*256). The division runs in 48 bits.
module q8_8_reciprocal #(parameter SUM_W = 32) (
    input  wire signed [SUM_W-1:0] sum,  // > 0 (sum of >= 1 nonzero Q8.8 exp terms)
    output wire signed [15:0] recip
);
    wire signed [47:0] sum_wide = sum;
    localparam signed [47:0] NUMERATOR = 48'sd65536;
    wire signed [47:0] recip_full = (sum == 0) ? 48'sd0 : (NUMERATOR / sum_wide);
    assign recip = recip_full[15:0];
endmodule
