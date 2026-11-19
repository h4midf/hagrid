// QUARK Eq.18: a/b = exp(ln(a)-ln(b)), a, b > 0. quark_appro_exp only accepts x <= 0,
// so when log_ratio = ln(a)-ln(b) > 0 (a > b) it uses exp(y) = 1/exp(-y) through
// quark_recip.
module quark_shared_div #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] a,   // a > 0 required
    input  wire signed [W-1:0] b,   // b > 0 required
    output wire signed [W-1:0] y    // approx a/b
);
    wire signed [W-1:0] ln_a, ln_b;
    quark_appro_ln #(.W(W), .FRAC(FRAC)) u_ln_a (.x(a), .y(ln_a));
    quark_appro_ln #(.W(W), .FRAC(FRAC)) u_ln_b (.x(b), .y(ln_b));

    wire signed [W:0] log_ratio_full = $signed(ln_a) - $signed(ln_b);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    wire signed [W-1:0] log_ratio = (log_ratio_full > MAXV) ? MAXV[W-1:0] :
                                     (log_ratio_full < MINV) ? MINV[W-1:0] : log_ratio_full[W-1:0];
    wire positive = ~log_ratio[W-1] & (|log_ratio); // log_ratio > 0

    wire signed [W-1:0] neg_log_ratio = -log_ratio;
    wire signed [W-1:0] exp_neg_branch, exp_pos_branch;
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_exp_neg (.x(log_ratio), .y(exp_neg_branch));       // valid when log_ratio<=0
    quark_appro_exp #(.W(W), .FRAC(FRAC)) u_exp_pos (.x(neg_log_ratio), .y(exp_pos_branch));   // valid when log_ratio>0 (neg_log_ratio<=0 then)

    wire signed [W-1:0] recip_pos;
    quark_recip #(.W(W), .FRAC(FRAC)) u_recip (.x(exp_pos_branch), .y(recip_pos));

    assign y = positive ? recip_pos : exp_neg_branch;
endmodule
