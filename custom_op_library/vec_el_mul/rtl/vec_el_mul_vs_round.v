// Elementwise vector*scalar multiply (cim.vec_el_mul's rhs_scalar case): one
// W-bit signed scalar is broadcast to all N lanes. Same round-then-shift
// rescale as vec_el_mul_vv_round.v.
module vec_el_mul_vs_round #(parameter N = 8, parameter W = 32, parameter FRAC = 20) (
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [W-1:0]   y_scalar,
    output wire signed [N*W-1:0] out_flat
);
    localparam signed [2*W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [2*W:0] MINV = -(1 <<< (W - 1));

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            wire signed [2*W-1:0] p    = $signed(x_flat[i*W +: W]) * $signed(y_scalar);
            wire signed [2*W-1:0] half = (FRAC == 0) ? {2*W{1'b0}} : ({{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1));
            wire signed [2*W-1:0] r    = (p + half) >>> FRAC;
            assign out_flat[i*W +: W] = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];
        end
    endgenerate
endmodule
