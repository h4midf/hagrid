// One element of e2softmax_row8.v's log2exp_q8_8 (SOLE's Eq.8:
// k_y = -floor(x + x/2 - x/16), a 4-bit log2-quantized exponent), on its own so a
// row of any width is costed per element after max_reduce_tree. x arrives already
// in Q8.8, so log2exp_standalone(d <<< 8) == log2exp_q8_8(d).
module log2exp_standalone (
    input  wire signed [23:0] x,  // Q8.8 (wide container), x <= 0, natural-log delta from the row max
    output reg [3:0] k_y,         // 4-bit log2-quantized exponent, clipped to [0,15]
    output wire [8:0] e_recon     // e2softmax_row8.v's recon: 256 >> k_y (0 if k_y > 8), for the row sum
);
    wire signed [23:0] xshift = x + (x >>> 1) - (x >>> 4);
    wire signed [23:0] k_raw  = xshift >>> 8;
    wire signed [23:0] k_neg  = -k_raw;
    always @* begin
        if (k_neg > 24'sd15)
            k_y = 4'd15;
        else if (k_neg < 24'sd0)
            k_y = 4'd0;
        else
            k_y = k_neg[3:0];
    end

    assign e_recon = (k_y > 4'd8) ? 9'd0 : (9'd256 >> k_y);
endmodule
