// I-BERT's integer-only Softmax (ICML 2021), Eq.10-14, Q8.8, row of 8. Each
// d_i <= 0 is range-reduced to d_i = -ln2*z_i + p_i with p_i in (-ln2, 0],
// exp(p_i) = 0.3585*(p_i+1.353)^2 + 0.344, and exp(d_i) = exp(p_i) >> z_i.
// Normalization is softmax_row8.v's one reciprocal and 8 multiplies.
module ibert_exp_pwl (
    input  signed [8:0] d,             // integer diff from row max, d <= 0, range -255..0
    output reg  signed [15:0] exp_out  // Q8.8 approximation of exp(d/256), in [0,1]
);
    localparam signed [15:0] LN2_Q8_8    = 16'sd177;  // round(ln2 * 256)
    localparam signed [15:0] INV_LN2_Q88 = 16'sd369;  // round(256 / ln2)
    localparam signed [15:0] C0_Q8_8 = 16'sd92;   // 0.3585
    localparam signed [15:0] C1_Q8_8 = 16'sd346;  // 1.353
    localparam signed [15:0] C2_Q8_8 = 16'sd88;   // 0.344
    localparam N_MAX = 9; // 2^-9 rounds to 0 in Q8.8

    wire signed [23:0] x       = {{15{d[8]}}, d} <<< 8; // Q8.8

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

module ibert_softmax_row8 (
    input  [7:0] s0, s1, s2, s3, s4, s5, s6, s7,
    output signed [15:0] p0, p1, p2, p3, p4, p5, p6, p7
);
    wire [7:0] max_val;
    max_reduce8 u_max (
        .s0(s0), .s1(s1), .s2(s2), .s3(s3),
        .s4(s4), .s5(s5), .s6(s6), .s7(s7),
        .max_out(max_val)
    );

    wire signed [8:0] d0 = {1'b0, s0} - {1'b0, max_val};
    wire signed [8:0] d1 = {1'b0, s1} - {1'b0, max_val};
    wire signed [8:0] d2 = {1'b0, s2} - {1'b0, max_val};
    wire signed [8:0] d3 = {1'b0, s3} - {1'b0, max_val};
    wire signed [8:0] d4 = {1'b0, s4} - {1'b0, max_val};
    wire signed [8:0] d5 = {1'b0, s5} - {1'b0, max_val};
    wire signed [8:0] d6 = {1'b0, s6} - {1'b0, max_val};
    wire signed [8:0] d7 = {1'b0, s7} - {1'b0, max_val};

    wire signed [15:0] e0, e1, e2, e3, e4, e5, e6, e7;
    ibert_exp_pwl u_e0 (.d(d0), .exp_out(e0));
    ibert_exp_pwl u_e1 (.d(d1), .exp_out(e1));
    ibert_exp_pwl u_e2 (.d(d2), .exp_out(e2));
    ibert_exp_pwl u_e3 (.d(d3), .exp_out(e3));
    ibert_exp_pwl u_e4 (.d(d4), .exp_out(e4));
    ibert_exp_pwl u_e5 (.d(d5), .exp_out(e5));
    ibert_exp_pwl u_e6 (.d(d6), .exp_out(e6));
    ibert_exp_pwl u_e7 (.d(d7), .exp_out(e7));

    wire signed [17:0] r01 = e0 + e1;
    wire signed [17:0] r23 = e2 + e3;
    wire signed [17:0] r45 = e4 + e5;
    wire signed [17:0] r67 = e6 + e7;
    wire signed [18:0] r0123 = r01 + r23;
    wire signed [18:0] r4567 = r45 + r67;
    wire signed [19:0] exp_sum = r0123 + r4567;

    wire signed [31:0] recip_full = (exp_sum == 0) ? 32'sd0 : (32'sd65536 / exp_sum);
    wire signed [15:0] recip = recip_full[15:0];

    wire signed [31:0] p0_full = e0 * recip;
    wire signed [31:0] p1_full = e1 * recip;
    wire signed [31:0] p2_full = e2 * recip;
    wire signed [31:0] p3_full = e3 * recip;
    wire signed [31:0] p4_full = e4 * recip;
    wire signed [31:0] p5_full = e5 * recip;
    wire signed [31:0] p6_full = e6 * recip;
    wire signed [31:0] p7_full = e7 * recip;

    assign p0 = p0_full[23:8];
    assign p1 = p1_full[23:8];
    assign p2 = p2_full[23:8];
    assign p3 = p3_full[23:8];
    assign p4 = p4_full[23:8];
    assign p5 = p5_full[23:8];
    assign p6 = p6_full[23:8];
    assign p7 = p7_full[23:8];
endmodule
