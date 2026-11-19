// Elementwise vector+vector add, N lanes of W-bit signed data, saturating on
// overflow (see ../../common/fixed_point.md). Each lane is an independent
// (W+1)-bit add (wide enough that the sum of two W-bit signed numbers can
// never itself overflow) followed by a clamp back to the W-bit word range.
module vec_add_vv #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [N*W-1:0] y_flat,
    output wire signed [N*W-1:0] out_flat
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));

    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            wire signed [W:0] s = $signed(x_flat[i*W +: W]) + $signed(y_flat[i*W +: W]);
            assign out_flat[i*W +: W] = (s > MAXV) ? MAXV[W-1:0] : (s < MINV) ? MINV[W-1:0] : s[W-1:0];
        end
    endgenerate
endmodule
