// RMSNorm over N lanes of W-bit data with a scalar gamma (no mean, no beta):
// ms = mean(x^2) (truncating >> log2(N)), s = sqrt(ms) = ms*rsqrt(ms),
// s_eps = s + eps (eps added after the sqrt), inv = 1/s_eps = rsqrt(s_eps^2),
// y[i] = round(round(x[i]*inv)*gamma). rsqrt(0) is undefined, so s is muxed to 0
// when ms == 0. W=32/FRAC=20 for Q12.20, W=16/FRAC=8 for Q8.8.
module rms_norm #(parameter N = 8, parameter W = 32, parameter FRAC = 20) (
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [W-1:0]   gamma,
    input  wire signed [W-1:0]   eps,
    output wire signed [N*W-1:0] y_flat
);
    localparam LEVELS = $clog2(N);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);

    genvar i;
    wire signed [W*N-1:0] sq_flat;

    generate
        for (i = 0; i < N; i = i + 1) begin : sqlane
            wire signed [2*W-1:0] p = $signed(x_flat[i*W +: W]) * $signed(x_flat[i*W +: W]);
            wire signed [2*W-1:0] r = (p + HALF) >>> FRAC;
            assign sq_flat[i*W +: W] = (r > MAXV) ? MAXV[W-1:0] : (r < MINV) ? MINV[W-1:0] : r[W-1:0];
        end
    endgenerate

    // ms = mean(x^2) = sum(sq) / N
    wire signed [W+LEVELS-1:0] sum_sq;
    reduce_tree #(.N(N), .W(W)) u_sum_sq (.x_flat(sq_flat), .sum(sum_sq));
    wire signed [W+LEVELS-1:0] ms_wide = sum_sq >>> LEVELS;
    wire signed [W-1:0] ms = ms_wide[W-1:0];
    wire ms_is_zero = (ms == {W{1'b0}});

    // s = sqrt(ms) = ms * rsqrt(ms), muxed to 0 when ms==0.
    wire signed [W-1:0] r1;
    generate
        if (W == 32 && FRAC == 20) begin : q12_20_r1
            rsqrt_q12_20 u_rsqrt1 (.val(ms), .rsqrt_out(r1));
        end else if (W == 16 && FRAC == 8) begin : q8_8_r1
            rsqrt_q8_8 u_rsqrt1 (.val(ms), .rsqrt_out(r1));
        end
    endgenerate

    wire signed [2*W-1:0] sp = $signed(ms) * $signed(r1);
    wire signed [2*W-1:0] sr = (sp + HALF) >>> FRAC;
    wire signed [W-1:0] s_raw = (sr > MAXV) ? MAXV[W-1:0] : (sr < MINV) ? MINV[W-1:0] : sr[W-1:0];
    wire signed [W-1:0] s = ms_is_zero ? {W{1'b0}} : s_raw;

    // s_eps = s + eps (always > 0: s >= 0, eps > 0).
    wire signed [W:0] s_eps_full = $signed(s) + $signed(eps);
    wire signed [W-1:0] s_eps = (s_eps_full > MAXV) ? MAXV[W-1:0] :
                                 (s_eps_full < MINV) ? MINV[W-1:0] : s_eps_full[W-1:0];

    // inv = 1/s_eps = rsqrt(s_eps^2).
    wire signed [2*W-1:0] vp = $signed(s_eps) * $signed(s_eps);
    wire signed [2*W-1:0] vr = (vp + HALF) >>> FRAC;
    wire signed [W-1:0] v2 = (vr > MAXV) ? MAXV[W-1:0] : (vr < MINV) ? MINV[W-1:0] : vr[W-1:0];
    wire v2_is_zero = (v2 == {W{1'b0}});

    wire signed [W-1:0] inv_raw;
    generate
        if (W == 32 && FRAC == 20) begin : q12_20_r2
            rsqrt_q12_20 u_rsqrt2 (.val(v2), .rsqrt_out(inv_raw));
        end else if (W == 16 && FRAC == 8) begin : q8_8_r2
            rsqrt_q8_8 u_rsqrt2 (.val(v2), .rsqrt_out(inv_raw));
        end
    endgenerate
    // v2 can round to 0 even though s_eps > 0 (a small s_eps^2 underflows), so the
    // same guard as ms_is_zero.
    wire signed [W-1:0] inv = v2_is_zero ? {W{1'b0}} : inv_raw;

    generate
        for (i = 0; i < N; i = i + 1) begin : normscale
            wire signed [2*W-1:0] pn = $signed(x_flat[i*W +: W]) * $signed(inv);
            wire signed [2*W-1:0] rn = (pn + HALF) >>> FRAC;
            wire signed [W-1:0] norm = (rn > MAXV) ? MAXV[W-1:0] : (rn < MINV) ? MINV[W-1:0] : rn[W-1:0];

            wire signed [2*W-1:0] pa = $signed(norm) * $signed(gamma);
            wire signed [2*W-1:0] ra = (pa + HALF) >>> FRAC;
            assign y_flat[i*W +: W] = (ra > MAXV) ? MAXV[W-1:0] : (ra < MINV) ? MINV[W-1:0] : ra[W-1:0];
        end
    endgenerate
endmodule
