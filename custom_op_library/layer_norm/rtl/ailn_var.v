// ailayernorm_stream.v's variance step (combinational), with E(x^2) and E(x)^2 kept wide:
// var = max(clamp(ex2 - round(ex^2 >> FRAC)), 1) (clamped to W bits only after the
// subtraction), var_eps = max(clamp(var + eps), 1) (rsqrt needs val > 0).
module ailn_var #(parameter W = 32, parameter FRAC = 20, parameter EW = 79) (
    input  wire signed [W-1:0]  ex,
    input  wire signed [EW-1:0] ex2,
    input  wire signed [W-1:0]  eps,
    output wire signed [W-1:0]  var_eps
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [EW:0] MAXE = MAXV;
    localparam signed [EW:0] MINE = MINV;
    localparam signed [2*W-1:0] HALF_FRAC = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    localparam signed [W-1:0] ONE_LSB = 1;

    wire signed [2*W-1:0] ex_sq_full = $signed(ex) * $signed(ex);
    wire signed [2*W-1:0] ex_sq = (ex_sq_full + HALF_FRAC) >>> FRAC;
    wire signed [EW:0] var_full = $signed(ex2) - $signed(ex_sq);
    wire signed [W-1:0] var_raw = (var_full > MAXE) ? MAXV[W-1:0] : (var_full < MINE) ? MINV[W-1:0] : var_full[W-1:0];
    wire signed [W-1:0] var_val = (var_raw < ONE_LSB) ? ONE_LSB : var_raw;
    wire signed [W:0] var_eps_full = $signed(var_val) + $signed(eps);
    wire signed [W-1:0] var_eps_clamped = (var_eps_full > MAXV) ? MAXV[W-1:0] :
                                           (var_eps_full < MINV) ? MINV[W-1:0] : var_eps_full[W-1:0];
    assign var_eps = (var_eps_clamped < ONE_LSB) ? ONE_LSB : var_eps_clamped;
endmodule
