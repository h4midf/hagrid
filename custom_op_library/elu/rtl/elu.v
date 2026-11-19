// ELU: y = x if x > 0, else ALPHA * (exp(x) - 1). ALPHA is a compile-time constant in
// the module's fixed-point format (set by the caller; e.g. 1.0 = 1 << FRAC).
// W=32/FRAC=20 uses exp_pwl_q12_20, W=16/FRAC=8 uses exp_pwl_q8_8 (softmax's exp_pwl16
// table). One rescale multiply, rounded as vec_el_mul_vv_round.v.
module elu #(parameter W = 32, parameter FRAC = 20, parameter signed [W-1:0] ALPHA = 0) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    localparam signed [2*W-1:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [2*W-1:0] MINV = -(1 <<< (W - 1));
    localparam signed [W-1:0] ONE = 1 <<< FRAC;

    wire signed [W-1:0] e;

    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            exp_pwl_q12_20 u_exp (.x(x), .exp_out(e));
        end else if (W == 16 && FRAC == 8) begin : q8_8
            exp_pwl_q8_8 u_exp (.x(x), .exp_out(e));
        end
    endgenerate

    wire signed [W-1:0] term = e - ONE;               // exp(x) - 1, in [-1, 0]
    wire signed [2*W-1:0] p    = term * ALPHA;
    wire signed [2*W-1:0] half = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    wire signed [2*W-1:0] r    = (p + half) >>> FRAC;
    wire signed [W-1:0] neg_branch = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];

    assign y = x[W-1] ? neg_branch : x;
endmodule
