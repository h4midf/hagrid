// Elementwise select, N lanes of W-bit signed data: out[i] = mask[i] ? x[i] : fill.
// One 2-to-1 multiplexer per lane. mask is one bit per lane; fill is one W-bit word
// shared by every lane (a runtime input, not a parameter). torch.where(cond, x, c) with
// a scalar c, e.g. GPT-2's causal mask (cond: a stored lower-triangular bit, fill: the
// most negative word of the format, which softmax turns into a zero weight).
module vec_select #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    input  wire        [N-1:0]   mask,
    input  wire signed [W-1:0]   fill,
    output wire signed [N*W-1:0] out_flat
);
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            assign out_flat[i*W +: W] = mask[i] ? x_flat[i*W +: W] : fill;
        end
    endgenerate
endmodule
