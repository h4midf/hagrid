// One lane of ailayernorm_stream.v (combinational): PTF quantize to int8 by the
// per-element exponent alpha, reconstruct x_shifted, DynamicCompress +
// Square&Decompress, and the normalize + affine
// y = clamp(round(round(clamp(x_shifted - ex) * inv_std >> FRAC) * gamma >> FRAC) + beta).
// The decompressed square is rescaled by 2*alpha + FRAC, to Q.FRAC units like E(x)^2,
// and kept at 2W bits (an outlier's square does not fit the data width).
module ailn_stream_lane #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0]   x,
    input  wire signed [7:0]     alpha,
    input  wire signed [W-1:0]   ex,
    input  wire signed [W-1:0]   inv_std,
    input  wire signed [W-1:0]   gamma,
    input  wire signed [W-1:0]   beta,
    output wire signed [W-1:0]   x_shifted,
    output wire signed [2*W-1:0] xc2_shifted,
    output wire signed [W-1:0]   y
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [2*W-1:0] MAXV2 = MAXV;
    localparam signed [2*W-1:0] MINV2 = MINV;
    localparam signed [2*W-1:0] HALF_FRAC = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);

    wire signed [15:0] alpha_w   = alpha;
    wire signed [15:0] frac_w    = FRAC;
    wire signed [15:0] shift_amt = alpha_w + frac_w;

    wire signed [2*W-1:0] x_wide = x;
    wire signed [2*W-1:0] half_dyn = (shift_amt > 16'sd0)
        ? ({{(2*W-1){1'b0}}, 1'b1} <<< (shift_amt - 16'sd1)) : {(2*W){1'b0}};
    wire signed [2*W-1:0] q_rshift = (shift_amt >= 16'sd0)
        ? ((x_wide + half_dyn) >>> shift_amt) : (x_wide <<< (-shift_amt));
    wire signed [2*W-1:0] MAXQ = 16'sd127;
    wire signed [2*W-1:0] MINQ = -16'sd128;
    wire signed [2*W-1:0] q_clip = (q_rshift > MAXQ) ? MAXQ : (q_rshift < MINQ) ? MINQ : q_rshift;
    wire signed [7:0] x_q = q_clip[7:0];

    wire signed [2*W-1:0] xq_wide = x_q;
    wire signed [2*W-1:0] half_dyn2 = (shift_amt < 16'sd0)
        ? ({{(2*W-1){1'b0}}, 1'b1} <<< ((-shift_amt) - 16'sd1)) : {(2*W){1'b0}};
    wire signed [2*W-1:0] x_shifted_wide = (shift_amt >= 16'sd0)
        ? (xq_wide <<< shift_amt) : ((xq_wide + half_dyn2) >>> (-shift_amt));
    assign x_shifted = (x_shifted_wide > MAXV2) ? MAXV[W-1:0] :
                       (x_shifted_wide < MINV2) ? MINV[W-1:0] : x_shifted_wide[W-1:0];

    wire [7:0] mag  = x_q[7] ? (-x_q) : x_q;
    wire       high = (mag >= 8'd64);
    wire [7:0] xc_raw = high ? (mag >> 4) : (mag >> 2);
    wire [3:0] xc = (xc_raw > 8'd15) ? 4'd15 : xc_raw[3:0];
    wire [7:0] xc2_raw = xc * xc;
    wire [11:0] xc2 = high ? ({4'b0, xc2_raw} <<< 4) : {4'b0, xc2_raw};

    // xc2 << (2*alpha + FRAC): the square in Q.FRAC units, kept at 2W bits
    wire signed [15:0] sq_shift = (alpha_w <<< 1) + frac_w;
    wire signed [2*W-1:0] xc2_s = {4'b0, xc2};
    wire signed [2*W-1:0] half_dyn3 = (sq_shift < 16'sd0)
        ? ({{(2*W-1){1'b0}}, 1'b1} <<< ((-sq_shift) - 16'sd1)) : {(2*W){1'b0}};
    assign xc2_shifted = (sq_shift >= 16'sd0) ? (xc2_s <<< sq_shift) : ((xc2_s + half_dyn3) >>> (-sq_shift));

    wire signed [W:0] diff_full = $signed(x_shifted) - $signed(ex);
    wire signed [W-1:0] diff = (diff_full > MAXV) ? MAXV[W-1:0] : (diff_full < MINV) ? MINV[W-1:0] : diff_full[W-1:0];
    wire signed [2*W-1:0] pn = $signed(diff) * $signed(inv_std);
    wire signed [2*W-1:0] rn = (pn + HALF_FRAC) >>> FRAC;
    wire signed [W-1:0] norm = (rn > MAXV2) ? MAXV[W-1:0] : (rn < MINV2) ? MINV[W-1:0] : rn[W-1:0];
    wire signed [2*W-1:0] pa = $signed(norm) * $signed(gamma);
    wire signed [2*W-1:0] ra = (pa + HALF_FRAC) >>> FRAC;
    wire signed [W-1:0] scaled = (ra > MAXV2) ? MAXV[W-1:0] : (ra < MINV2) ? MINV[W-1:0] : ra[W-1:0];
    wire signed [W:0] out_full = $signed(scaled) + $signed(beta);
    assign y = (out_full > MAXV) ? MAXV[W-1:0] : (out_full < MINV) ? MINV[W-1:0] : out_full[W-1:0];
endmodule
