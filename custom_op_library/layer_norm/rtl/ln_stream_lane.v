// One lane of layer_norm_stream.v (combinational): d = clamp(x - mean),
// r1 = round(d * (norm ? inv_std : d) >> FRAC); sq = r1 at 2W bits, unclamped (the
// square in VAR: a square does not fit the data width for an outlier element, e.g.
// |d| > 45 at Q12.20); m1 = clamp(r1) (the normalized value in NORM);
// y = clamp(round(m1 * gamma >> FRAC) + beta). One multiplier shared by the square and
// the normalize.
module ln_stream_lane #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0]   x,
    input  wire signed [W-1:0]   mean,
    input  wire signed [W-1:0]   inv_std,
    input  wire                  norm,
    input  wire signed [W-1:0]   gamma,
    input  wire signed [W-1:0]   beta,
    output wire signed [2*W-1:0] sq,
    output wire signed [W-1:0]   y
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);

    wire signed [W:0] d_full = $signed(x) - $signed(mean);
    wire signed [W-1:0] d = (d_full > MAXV) ? MAXV[W-1:0] : (d_full < MINV) ? MINV[W-1:0] : d_full[W-1:0];
    wire signed [W-1:0] b = norm ? inv_std : d;
    wire signed [2*W-1:0] p1 = $signed(d) * $signed(b);
    wire signed [2*W-1:0] r1 = (p1 + HALF) >>> FRAC;
    assign sq = r1;
    wire signed [W-1:0] m1 = (r1 > MAXV) ? MAXV[W-1:0] : (r1 < MINV) ? MINV[W-1:0] : r1[W-1:0];

    wire signed [2*W-1:0] p2 = $signed(m1) * $signed(gamma);
    wire signed [2*W-1:0] r2 = (p2 + HALF) >>> FRAC;
    wire signed [W-1:0] scaled = (r2 > MAXV) ? MAXV[W-1:0] : (r2 < MINV) ? MINV[W-1:0] : r2[W-1:0];
    wire signed [W:0] y_full = $signed(scaled) + $signed(beta);
    assign y = (y_full > MAXV) ? MAXV[W-1:0] : (y_full < MINV) ? MINV[W-1:0] : y_full[W-1:0];
endmodule
