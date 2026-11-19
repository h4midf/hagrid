// SOLE's E2Softmax (ICCAD 2023), Eq.7-13, Q8.8, row of 8. Each exp is log2-quantized
// to a 4-bit exponent k_y, e_i = 2^-k_y is summed, and ALDivision
// (S = 2^k_s*(1+s), a bias-corrected 1-bit linear 1/(1+s)) replaces the divider.
// The global max and sum replace Algorithm 1's running ones (exact for a whole row).
// Uses softmax_row8.v's max_reduce8. Matches run_scripts/measure_accuracy/
// sole_bert_large.py.
module log2exp_q8_8 (
    input  signed [8:0] d,             // raw score diff, d <= 0, range -255..0
    output reg  [3:0] k_y              // 4-bit log2-quantized exponent, clipped to [0,15]
);
    wire signed [23:0] x = {{15{d[8]}}, d} <<< 8; // Q8.8
    // xshift = x + x/2 - x/16 = x*1.4375 (Eq.8), <= 0
    wire signed [23:0] xshift = x + (x >>> 1) - (x >>> 4);
    wire signed [23:0] k_raw  = xshift >>> 8; // floor division by 256
    wire signed [23:0] k_neg  = -k_raw;       // >= 0
    always @* begin
        if (k_neg > 24'sd15)
            k_y = 4'd15;
        else if (k_neg < 24'sd0)
            k_y = 4'd0;
        else
            k_y = k_neg[3:0];
    end
endmodule

module aldivision_q8_8 (
    input  [3:0] k_y,                   // from log2exp_q8_8, in [0,15]
    input  signed [19:0] s_sum,         // Q8.8 row sum, > 0 (sum of >=1 nonzero exp terms)
    output reg signed [15:0] p_out      // Q8.8 approximation of 2^-k_y / S, in [0,1]
);
    function [4:0] msb_pos;
        input [19:0] v;
        integer i;
        reg found;
        begin
            msb_pos = 5'd0;
            found = 1'b0;
            for (i = 19; i >= 0; i = i - 1) begin
                if (!found && v[i]) begin
                    msb_pos = i[4:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [19:0] s_u      = s_sum[19:0]; // s_sum > 0 by contract, safe to treat as unsigned
    wire [4:0]  p_bit     = msb_pos(s_u);
    wire signed [6:0] k_s = {2'b00, p_bit} - 7'sd8; // p - FRAC
    wire next_bit = (p_bit == 5'd0) ? 1'b0 : s_u[p_bit - 1'b1];

    localparam signed [15:0] CONST_HI = 16'sd419; // round(1.636*256), q=0
    localparam signed [15:0] CONST_LO = 16'sd291; // round(1.136*256), q=0.5
    wire signed [15:0] const_sel = next_bit ? CONST_LO : CONST_HI;

    // Widened by plain assignment, not concatenation: a concatenation is unsigned
    // and would make the whole sum unsigned.
    wire signed [8:0] k_y_w = k_y;
    wire signed [8:0] k_s_w = k_s;
    wire signed [8:0] ks_sum = k_y_w + k_s_w + 9'sd1;

    always @* begin
        if (ks_sum >= 9'sd0) begin
            if (ks_sum >= 9'sd16)
                p_out = 16'sd0; // shifted entirely out of Q8.8 range
            else
                p_out = const_sel >>> ks_sum[3:0];
        end else begin
            // negative ks_sum is a left shift; saturate when it would overflow Q8.8
            if (ks_sum <= -9'sd16)
                p_out = 16'sd32767;
            else
                p_out = const_sel <<< (-ks_sum[3:0]);
        end
    end
endmodule

module e2softmax_row8 (
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

    wire [3:0] k0, k1, k2, k3, k4, k5, k6, k7;
    log2exp_q8_8 u_k0 (.d(d0), .k_y(k0));
    log2exp_q8_8 u_k1 (.d(d1), .k_y(k1));
    log2exp_q8_8 u_k2 (.d(d2), .k_y(k2));
    log2exp_q8_8 u_k3 (.d(d3), .k_y(k3));
    log2exp_q8_8 u_k4 (.d(d4), .k_y(k4));
    log2exp_q8_8 u_k5 (.d(d5), .k_y(k5));
    log2exp_q8_8 u_k6 (.d(d6), .k_y(k6));
    log2exp_q8_8 u_k7 (.d(d7), .k_y(k7));

    // e_i = 256 >> k_i (0 if k_i > 8, beyond Q8.8's own resolution)
    function [8:0] recon;
        input [3:0] k;
        begin
            recon = (k > 4'd8) ? 9'd0 : (9'd256 >> k);
        end
    endfunction
    wire [8:0] e0 = recon(k0);
    wire [8:0] e1 = recon(k1);
    wire [8:0] e2 = recon(k2);
    wire [8:0] e3 = recon(k3);
    wire [8:0] e4 = recon(k4);
    wire [8:0] e5 = recon(k5);
    wire [8:0] e6 = recon(k6);
    wire [8:0] e7 = recon(k7);

    wire [10:0] r01 = e0 + e1;
    wire [10:0] r23 = e2 + e3;
    wire [10:0] r45 = e4 + e5;
    wire [10:0] r67 = e6 + e7;
    wire [11:0] r0123 = r01 + r23;
    wire [11:0] r4567 = r45 + r67;
    wire [12:0] s_sum_u = r0123 + r4567; // > 0 always: at least one e_i >= 1
    wire signed [19:0] s_sum = s_sum_u; // zero-extends

    aldivision_q8_8 u_p0 (.k_y(k0), .s_sum(s_sum), .p_out(p0));
    aldivision_q8_8 u_p1 (.k_y(k1), .s_sum(s_sum), .p_out(p1));
    aldivision_q8_8 u_p2 (.k_y(k2), .s_sum(s_sum), .p_out(p2));
    aldivision_q8_8 u_p3 (.k_y(k3), .s_sum(s_sum), .p_out(p3));
    aldivision_q8_8 u_p4 (.k_y(k4), .s_sum(s_sum), .p_out(p4));
    aldivision_q8_8 u_p5 (.k_y(k5), .s_sum(s_sum), .p_out(p5));
    aldivision_q8_8 u_p6 (.k_y(k6), .s_sum(s_sum), .p_out(p6));
    aldivision_q8_8 u_p7 (.k_y(k7), .s_sum(s_sum), .p_out(p7));
endmodule
