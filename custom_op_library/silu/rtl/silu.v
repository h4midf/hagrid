// SiLU: y = x * sigmoid(x). exp_pwl only takes arguments <= 0, so a sign mux picks
// sigmoid(x) = exp(x)/(1+exp(x)) for x < 0 and 1/(1+exp(-x)) for x >= 0. One
// division gives sigmoid, one rescale-multiply gives x*sigmoid(x).
// W=32/FRAC=20 uses exp_pwl_q12_20, W=16/FRAC=8 uses exp_pwl_q8_8.
module silu #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    localparam signed [W-1:0]   SCALE = 1 <<< FRAC;
    localparam signed [2*W-1:0] MAXV  = (1 <<< (W - 1)) - 1;
    localparam signed [2*W-1:0] MINV  = -(1 <<< (W - 1));

    wire signed [W-1:0] exp_arg = x[W-1] ? x : -x;
    wire signed [W-1:0] e;

    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            exp_pwl_q12_20 u_exp (.x(exp_arg), .exp_out(e));
        end else if (W == 16 && FRAC == 8) begin : q8_8
            exp_pwl_q8_8 u_exp (.x(exp_arg), .exp_out(e));
        end
    endgenerate

    wire signed [W-1:0] denom   = SCALE + e;          // > 0
    wire signed [W-1:0] sig_num = x[W-1] ? e : SCALE;  // >= 0

    wire signed [2*W-1:0] sig_full = ($signed(sig_num) * $signed(SCALE)) / $signed(denom);
    wire signed [W-1:0] sig_q      = sig_full[W-1:0];  // sigmoid(x), in [0, 1]

    wire signed [2*W-1:0] p    = x * sig_q;
    wire signed [2*W-1:0] half = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    wire signed [2*W-1:0] r    = (p + half) >>> FRAC;
    assign y = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];
endmodule
