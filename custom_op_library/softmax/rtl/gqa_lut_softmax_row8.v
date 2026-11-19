// GQA-LUT (DAC'24) softmax: softmax_row8.v's datapath (max_reduce8, per-lane exp,
// one shared reciprocal) with the exp replaced by gqa_lut_exp8. GQA-LUT has no
// division trick, so normalization is softmax_row8.v's divide.
//
// d <<< 8 ranges -65280..0, beyond int16, so it saturates to -32768 before
// gqa_lut_exp8. Both are far below DOM_LO (-8.0), which outputs 0, so the result
// is the same.
module gqa_lut_softmax_row8 (
    input  [7:0] s0, s1, s2, s3, s4, s5, s6, s7, // 8-bit attention scores
    output signed [15:0] p0, p1, p2, p3, p4, p5, p6, p7 // Q8.8, each in [0,1], sum to ~1.0
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

    wire signed [23:0] x0_wide = {{15{d0[8]}}, d0} <<< 8;
    wire signed [23:0] x1_wide = {{15{d1[8]}}, d1} <<< 8;
    wire signed [23:0] x2_wide = {{15{d2[8]}}, d2} <<< 8;
    wire signed [23:0] x3_wide = {{15{d3[8]}}, d3} <<< 8;
    wire signed [23:0] x4_wide = {{15{d4[8]}}, d4} <<< 8;
    wire signed [23:0] x5_wide = {{15{d5[8]}}, d5} <<< 8;
    wire signed [23:0] x6_wide = {{15{d6[8]}}, d6} <<< 8;
    wire signed [23:0] x7_wide = {{15{d7[8]}}, d7} <<< 8;

    localparam signed [23:0] INT16_MIN_WIDE = -24'sd32768;
    wire signed [15:0] x0 = (x0_wide < INT16_MIN_WIDE) ? -16'sd32768 : x0_wide[15:0];
    wire signed [15:0] x1 = (x1_wide < INT16_MIN_WIDE) ? -16'sd32768 : x1_wide[15:0];
    wire signed [15:0] x2 = (x2_wide < INT16_MIN_WIDE) ? -16'sd32768 : x2_wide[15:0];
    wire signed [15:0] x3 = (x3_wide < INT16_MIN_WIDE) ? -16'sd32768 : x3_wide[15:0];
    wire signed [15:0] x4 = (x4_wide < INT16_MIN_WIDE) ? -16'sd32768 : x4_wide[15:0];
    wire signed [15:0] x5 = (x5_wide < INT16_MIN_WIDE) ? -16'sd32768 : x5_wide[15:0];
    wire signed [15:0] x6 = (x6_wide < INT16_MIN_WIDE) ? -16'sd32768 : x6_wide[15:0];
    wire signed [15:0] x7 = (x7_wide < INT16_MIN_WIDE) ? -16'sd32768 : x7_wide[15:0];

    wire signed [15:0] e0, e1, e2, e3, e4, e5, e6, e7;
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e0 (.x(x0), .y(e0));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e1 (.x(x1), .y(e1));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e2 (.x(x2), .y(e2));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e3 (.x(x3), .y(e3));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e4 (.x(x4), .y(e4));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e5 (.x(x5), .y(e5));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e6 (.x(x6), .y(e6));
    gqa_lut_exp8 #(.W(16), .FRAC(8)) u_e7 (.x(x7), .y(e7));

    wire signed [17:0] r01 = e0 + e1;
    wire signed [17:0] r23 = e2 + e3;
    wire signed [17:0] r45 = e4 + e5;
    wire signed [17:0] r67 = e6 + e7;
    wire signed [18:0] r0123 = r01 + r23;
    wire signed [18:0] r4567 = r45 + r67;
    wire signed [19:0] exp_sum = r0123 + r4567; // > 0: the row max lane gives 256

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
