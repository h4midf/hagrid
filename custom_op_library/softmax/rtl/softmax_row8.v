// Softmax over one row of 8 elements, Q8.8. The exp is RMSL's base-2 decomposition:
// a shift for the integer power of 2 and a 16-segment piecewise-linear function for
// the fraction. A 4th-order Taylor series (AESHA/HyFlexPIM) was tried first and
// removed: it diverges away from 0 (Taylor4(-3) = 1.375 vs exp(-3) = 0.050).
// CMOS logic, not memory: the 16 (OFFSET, SLOPE) pairs are hardwired constants
// selected by a 16-way mux into one multiply-add.

module exp_pwl16 (
    input  signed [23:0] x,       // Q8.8 (in a wide container), x <= 0, natural-log delta from the row max
    output reg  signed [15:0] exp_out // Q8.8 approximation of exp(x), in [0, 1]
);
    localparam signed [15:0] LOG2E_Q8_8 = 16'sd369;  // round(log2(e) * 256): exp(x) = 2^(x*log2(e))

    // OFFSET[k] = round(2^(-k/16) * 256), SLOPE[k] = round((2^(-(k+1)/16) - 2^(-k/16)) * 256);
    // max error 1 Q8.8 LSB.
    localparam N_MAX = 9; // 2^-9 rounds to 0 in Q8.8

    wire signed [39:0] y_full = x * LOG2E_Q8_8;      // Q16.16-scale product
    wire signed [31:0] y      = y_full[39:8];        // rescale to Q8.8
    wire signed [31:0] neg_y  = -y;                  // >= 0, since x <= 0
    wire [23:0] n_raw   = neg_y[31:8];                // integer part (number of halvings)
    wire [7:0]  r_frac  = neg_y[7:0];                 // fractional part, scaled by 256
    wire [3:0]  seg      = r_frac[7:4];
    wire [3:0]  intra    = r_frac[3:0];

    reg signed [9:0] off;  // 10 bits: OFFSET[0] = 256
    reg signed [4:0] slope;
    always @* begin
        case (seg)
            4'd0:  begin off = 10'sd256; slope = -5'sd11; end
            4'd1:  begin off = 9'sd245; slope = -5'sd10; end
            4'd2:  begin off = 9'sd235; slope = -5'sd10; end
            4'd3:  begin off = 9'sd225; slope = -5'sd10; end
            4'd4:  begin off = 9'sd215; slope = -5'sd9;  end
            4'd5:  begin off = 9'sd206; slope = -5'sd9;  end
            4'd6:  begin off = 9'sd197; slope = -5'sd8;  end
            4'd7:  begin off = 9'sd189; slope = -5'sd8;  end
            4'd8:  begin off = 9'sd181; slope = -5'sd8;  end
            4'd9:  begin off = 9'sd173; slope = -5'sd7;  end
            4'd10: begin off = 9'sd166; slope = -5'sd7;  end
            4'd11: begin off = 9'sd159; slope = -5'sd7;  end
            4'd12: begin off = 9'sd152; slope = -5'sd6;  end
            4'd13: begin off = 9'sd146; slope = -5'sd6;  end
            4'd14: begin off = 9'sd140; slope = -5'sd6;  end
            4'd15: begin off = 9'sd134; slope = -5'sd6;  end
        endcase
    end

    wire signed [15:0] g_full = (off <<< 4) + slope * $signed({1'b0, intra});
    wire signed [11:0] g_r    = g_full[15:4]; // Q8.8 approximation of 2^(-r), in [128, 256]

    always @* begin
        if (n_raw >= N_MAX)
            exp_out = 16'sd0;
        else
            exp_out = $signed(g_r) >>> n_raw[3:0];
    end
endmodule

module max_reduce8 (
    input  [7:0] s0, s1, s2, s3, s4, s5, s6, s7,
    output [7:0] max_out
);
    wire [7:0] m01 = (s0 > s1) ? s0 : s1;
    wire [7:0] m23 = (s2 > s3) ? s2 : s3;
    wire [7:0] m45 = (s4 > s5) ? s4 : s5;
    wire [7:0] m67 = (s6 > s7) ? s6 : s7;
    wire [7:0] m0123 = (m01 > m23) ? m01 : m23;
    wire [7:0] m4567 = (m45 > m67) ? m45 : m67;
    assign max_out = (m0123 > m4567) ? m0123 : m4567;
endmodule

module softmax_row8 (
    input  [7:0] s0, s1, s2, s3, s4, s5, s6, s7, // 8-bit attention scores
    output signed [15:0] p0, p1, p2, p3, p4, p5, p6, p7 // Q8.8, each in [0,1], sum to ~1.0
);
    wire [7:0] max_val;
    max_reduce8 u_max (
        .s0(s0), .s1(s1), .s2(s2), .s3(s3),
        .s4(s4), .s5(s5), .s6(s6), .s7(s7),
        .max_out(max_val)
    );

    // signed integer difference from max (always <= 0, range -255..0)
    wire signed [8:0] d0 = {1'b0, s0} - {1'b0, max_val};
    wire signed [8:0] d1 = {1'b0, s1} - {1'b0, max_val};
    wire signed [8:0] d2 = {1'b0, s2} - {1'b0, max_val};
    wire signed [8:0] d3 = {1'b0, s3} - {1'b0, max_val};
    wire signed [8:0] d4 = {1'b0, s4} - {1'b0, max_val};
    wire signed [8:0] d5 = {1'b0, s5} - {1'b0, max_val};
    wire signed [8:0] d6 = {1'b0, s6} - {1'b0, max_val};
    wire signed [8:0] d7 = {1'b0, s7} - {1'b0, max_val};

    // To Q8.8. No clamp: exp_pwl16 saturates large negative x to 0.
    wire signed [23:0] x0 = {{15{d0[8]}}, d0} <<< 8;
    wire signed [23:0] x1 = {{15{d1[8]}}, d1} <<< 8;
    wire signed [23:0] x2 = {{15{d2[8]}}, d2} <<< 8;
    wire signed [23:0] x3 = {{15{d3[8]}}, d3} <<< 8;
    wire signed [23:0] x4 = {{15{d4[8]}}, d4} <<< 8;
    wire signed [23:0] x5 = {{15{d5[8]}}, d5} <<< 8;
    wire signed [23:0] x6 = {{15{d6[8]}}, d6} <<< 8;
    wire signed [23:0] x7 = {{15{d7[8]}}, d7} <<< 8;

    wire signed [15:0] e0, e1, e2, e3, e4, e5, e6, e7;
    exp_pwl16 u_e0 (.x(x0), .exp_out(e0));
    exp_pwl16 u_e1 (.x(x1), .exp_out(e1));
    exp_pwl16 u_e2 (.x(x2), .exp_out(e2));
    exp_pwl16 u_e3 (.x(x3), .exp_out(e3));
    exp_pwl16 u_e4 (.x(x4), .exp_out(e4));
    exp_pwl16 u_e5 (.x(x5), .exp_out(e5));
    exp_pwl16 u_e6 (.x(x6), .exp_out(e6));
    exp_pwl16 u_e7 (.x(x7), .exp_out(e7));

    wire signed [17:0] r01 = e0 + e1;
    wire signed [17:0] r23 = e2 + e3;
    wire signed [17:0] r45 = e4 + e5;
    wire signed [17:0] r67 = e6 + e7;
    wire signed [18:0] r0123 = r01 + r23;
    wire signed [18:0] r4567 = r45 + r67;
    wire signed [19:0] exp_sum = r0123 + r4567; // Q8.8 + headroom, > 0 (sum of 8 non-negative terms, at least one = 256)

    // One reciprocal of the row sum, recip = 65536 / exp_sum, shared by all 8 multiplies.
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
