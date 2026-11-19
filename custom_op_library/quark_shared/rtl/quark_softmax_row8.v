// QUARK's softmax (Eq.2's log-sum-exp form): x' = x - max(x), S = sum(exp(x')),
// z = x' - ln(S), p = exp(z) = exp(x')/S. z <= 0 always (S >= 1, since the max element
// contributes exp(0)), so appro_exp's x <= 0 domain holds without a clamp.
// Row width fixed at 8. Uses quark_appro_exp, quark_appro_ln and reduce_tree.
module quark_softmax_row8 #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] s0, s1, s2, s3, s4, s5, s6, s7,
    output wire signed [W-1:0] p0, p1, p2, p3, p4, p5, p6, p7
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));

    function signed [W-1:0] max2;
        input signed [W-1:0] a, b;
        begin
            max2 = (a > b) ? a : b;
        end
    endfunction
    wire signed [W-1:0] m01 = max2(s0, s1);
    wire signed [W-1:0] m23 = max2(s2, s3);
    wire signed [W-1:0] m45 = max2(s4, s5);
    wire signed [W-1:0] m67 = max2(s6, s7);
    wire signed [W-1:0] m0123 = max2(m01, m23);
    wire signed [W-1:0] m4567 = max2(m45, m67);
    wire signed [W-1:0] max_w = max2(m0123, m4567);

    wire signed [W:0] d0_f = $signed(s0) - $signed(max_w);
    wire signed [W:0] d1_f = $signed(s1) - $signed(max_w);
    wire signed [W:0] d2_f = $signed(s2) - $signed(max_w);
    wire signed [W:0] d3_f = $signed(s3) - $signed(max_w);
    wire signed [W:0] d4_f = $signed(s4) - $signed(max_w);
    wire signed [W:0] d5_f = $signed(s5) - $signed(max_w);
    wire signed [W:0] d6_f = $signed(s6) - $signed(max_w);
    wire signed [W:0] d7_f = $signed(s7) - $signed(max_w);
    wire signed [W-1:0] d0 = (d0_f < MINV) ? MINV[W-1:0] : d0_f[W-1:0];
    wire signed [W-1:0] d1 = (d1_f < MINV) ? MINV[W-1:0] : d1_f[W-1:0];
    wire signed [W-1:0] d2 = (d2_f < MINV) ? MINV[W-1:0] : d2_f[W-1:0];
    wire signed [W-1:0] d3 = (d3_f < MINV) ? MINV[W-1:0] : d3_f[W-1:0];
    wire signed [W-1:0] d4 = (d4_f < MINV) ? MINV[W-1:0] : d4_f[W-1:0];
    wire signed [W-1:0] d5 = (d5_f < MINV) ? MINV[W-1:0] : d5_f[W-1:0];
    wire signed [W-1:0] d6 = (d6_f < MINV) ? MINV[W-1:0] : d6_f[W-1:0];
    wire signed [W-1:0] d7 = (d7_f < MINV) ? MINV[W-1:0] : d7_f[W-1:0];

    wire signed [W-1:0] e0, e1, e2, e3, e4, e5, e6, e7;
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e0 (.x(d0), .y(e0));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e1 (.x(d1), .y(e1));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e2 (.x(d2), .y(e2));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e3 (.x(d3), .y(e3));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e4 (.x(d4), .y(e4));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e5 (.x(d5), .y(e5));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e6 (.x(d6), .y(e6));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_e7 (.x(d7), .y(e7));

    wire signed [8*W-1:0] e_flat = {e7, e6, e5, e4, e3, e2, e1, e0};
    wire signed [W+2:0] s_sum;
    reduce_tree #(.N(8), .W(W)) u_sum (.x_flat(e_flat), .sum(s_sum));
    localparam signed [W+2:0] ONE_LSB3 = 1;
    wire signed [W-1:0] s_sum_clamped = (s_sum < ONE_LSB3) ? ONE_LSB3[W-1:0] : s_sum[W-1:0]; // guard ln's x>0 contract

    wire signed [W-1:0] ln_s;
    quark_appro_ln #(.W(W), .FRAC(FRAC)) u_ln_s (.x(s_sum_clamped), .y(ln_s));

    localparam signed [W:0] ZERO_W1 = 0;
    function signed [W-1:0] zclamp;
        input signed [W:0] zf;
        begin
            zclamp = (zf < MINV) ? MINV[W-1:0] : (zf > ZERO_W1) ? {W{1'b0}} : zf[W-1:0];
        end
    endfunction
    wire signed [W:0] z0f = $signed(d0) - $signed(ln_s);
    wire signed [W:0] z1f = $signed(d1) - $signed(ln_s);
    wire signed [W:0] z2f = $signed(d2) - $signed(ln_s);
    wire signed [W:0] z3f = $signed(d3) - $signed(ln_s);
    wire signed [W:0] z4f = $signed(d4) - $signed(ln_s);
    wire signed [W:0] z5f = $signed(d5) - $signed(ln_s);
    wire signed [W:0] z6f = $signed(d6) - $signed(ln_s);
    wire signed [W:0] z7f = $signed(d7) - $signed(ln_s);
    wire signed [W-1:0] z0 = zclamp(z0f);
    wire signed [W-1:0] z1 = zclamp(z1f);
    wire signed [W-1:0] z2 = zclamp(z2f);
    wire signed [W-1:0] z3 = zclamp(z3f);
    wire signed [W-1:0] z4 = zclamp(z4f);
    wire signed [W-1:0] z5 = zclamp(z5f);
    wire signed [W-1:0] z6 = zclamp(z6f);
    wire signed [W-1:0] z7 = zclamp(z7f);

    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p0 (.x(z0), .y(p0));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p1 (.x(z1), .y(p1));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p2 (.x(z2), .y(p2));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p3 (.x(z3), .y(p3));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p4 (.x(z4), .y(p4));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p5 (.x(z5), .y(p5));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p6 (.x(z6), .y(p6));
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_p7 (.x(z7), .y(p7));
endmodule
