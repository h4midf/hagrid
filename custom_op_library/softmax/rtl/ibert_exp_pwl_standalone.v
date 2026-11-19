// One element of ibert_softmax_row8.v's ibert_exp_pwl (I-BERT Eq.10-14,
// range-reduced quadratic exp), on its own so a row of any width is costed per
// element after max_reduce_tree. x arrives already in Q8.8, so
// ibert_exp_pwl_standalone(d <<< 8) == ibert_exp_pwl(d).
module ibert_exp_pwl_standalone (
    input  wire signed [23:0] x,       // Q8.8 (wide container), x <= 0, natural-log delta from the row max
    output reg  signed [15:0] exp_out  // Q8.8 approximation of exp(x/256), in [0,1]
);
    localparam signed [15:0] LN2_Q8_8    = 16'sd177;  // round(ln2 * 256)
    localparam signed [15:0] INV_LN2_Q88 = 16'sd369;  // round(256 / ln2)
    localparam signed [15:0] C0_Q8_8 = 16'sd92;   // 0.3585
    localparam signed [15:0] C1_Q8_8 = 16'sd346;  // 1.353
    localparam signed [15:0] C2_Q8_8 = 16'sd88;   // 0.344
    localparam N_MAX = 9; // 2^-9 rounds to 0 in Q8.8

    // z = floor(-x / ln2), a plain integer: the Q16.16 product >> 16.
    wire signed [23:0] negx    = -x;
    wire signed [47:0] zq_full = negx * INV_LN2_Q88;
    wire signed [23:0] z       = zq_full >>> 16;

    // p = x + ln2*z, in (-ln2, 0]; z is an integer, so the product is already Q8.8.
    wire signed [39:0] lnz = LN2_Q8_8 * z;
    wire signed [23:0] p   = x + lnz[23:0];

    // exp_p = round(0.3585*(p+1.353)^2 + 0.344) in Q8.8. p fits 12 bits.
    localparam TW = 12;
    wire signed [TW-1:0] p_t    = p[TW-1:0];
    wire signed [TW-1:0] term   = p_t + C1_Q8_8[TW-1:0];
    wire signed [2*TW-1:0] sq_full  = term * term;
    wire signed [2*TW-1:0] sq_r     = (sq_full + (1 <<< 7)) >>> 8; // Q8.8
    wire signed [TW-1:0] sq         = sq_r[TW-1:0];
    wire signed [TW+9:0] c0sq_full  = C0_Q8_8 * sq;
    wire signed [TW+9:0] c0sq_r     = (c0sq_full + (1 <<< 7)) >>> 8;
    wire signed [23:0] exp_p        = c0sq_r + C2_Q8_8;

    always @* begin
        if (z >= N_MAX)
            exp_out = 16'sd0;
        else
            exp_out = exp_p[15:0] >>> z[3:0];
    end
endmodule
