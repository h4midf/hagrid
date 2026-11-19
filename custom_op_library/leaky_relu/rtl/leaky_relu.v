// LeakyReLU: y = x if x >= 0, else ALPHA * x. ALPHA is the negative_slope as a
// fixed-point integer at FRAC (round(0.01 * 2^20) = 10486 at Q12.20, round(0.01 * 2^8)
// = 3 at Q8.8), a parameter, so synthesis constant-folds the multiply into a shift-add
// network. W=32/FRAC=20 (Q12.20) or W=16/FRAC=8 (Q8.8).
module leaky_relu #(parameter W = 32, parameter FRAC = 20, parameter signed [W-1:0] ALPHA = 0) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    localparam signed [2*W-1:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [2*W-1:0] MINV = -(1 <<< (W - 1));

    wire signed [2*W-1:0] p    = x * ALPHA;
    wire signed [2*W-1:0] half = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    wire signed [2*W-1:0] r    = (p + half) >>> FRAC;
    wire signed [W-1:0] neg_branch = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];

    assign y = x[W-1] ? neg_branch : x;
endmodule
