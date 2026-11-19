// Elementwise vector*vector multiply, N lanes of W-bit signed fixed point with FRAC
// fraction bits. Each lane takes the 2W-bit product, adds half and shifts right by
// FRAC (round to nearest), then saturates to W bits.
module vec_el_mul_vv_round #(parameter N = 8, parameter W = 32, parameter FRAC = 20) (
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [N*W-1:0] y_flat,
    output wire signed [N*W-1:0] out_flat
);
    localparam signed [2*W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [2*W:0] MINV = -(1 <<< (W - 1));

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            wire signed [2*W-1:0] p    = $signed(x_flat[i*W +: W]) * $signed(y_flat[i*W +: W]);
            wire signed [2*W-1:0] half = (FRAC == 0) ? {2*W{1'b0}} : ({{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1));
            wire signed [2*W-1:0] r    = (p + half) >>> FRAC;
            assign out_flat[i*W +: W] = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];
        end
    endgenerate
endmodule
