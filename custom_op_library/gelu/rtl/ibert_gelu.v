// I-BERT's integer-only GELU (ICML 2021), Eq.4-9: GELU(x) = x*0.5*(1+erf(x/sqrt(2))),
// erf approximated by a clipped quadratic L(u) = sgn(u)*(a*(clip(|u|,max=-b)+b)^2+c),
// a=-0.2888, b=-1.769, c=1, applied to u = x/sqrt(2) (the clip is on u, as the paper's
// erf(x/sqrt(2))). sign(u) = sign(x), so the sign mux uses x's sign bit, with x = 0
// special-cased (a*term^2+c != 0 there). Two generate branches (Q12.20, Q8.8), the
// constants (1/sqrt(2), -b, a) at each width's scale.
//
// Each intermediate is sized to its own bound: term in [-NEG_B, 0], sq (term^2,
// rescaled) in [0, NEG_B_real^2], poly (a*term^2+c) in [~0.096, 1.0], one_plus_l in
// [0, 2.0]. Wider intermediates (W or 2W) make ABC's mapping of the chained multipliers
// stall.
//
// The saturate comparisons widen MAXV / MINV with a `localparam signed [N:0] WIDE =
// NARROW;` assignment, not `{...}`: a concatenation is unsigned, which would make the
// comparison unsigned.
module ibert_gelu #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            localparam signed [31:0] ONE        = 32'sd1048576;           // 1 * 2^20
            localparam signed [20:0] INV_SQRT2  = 21'sd741455;            // round(0.70710678 * 2^20), fits 21 bits
            localparam signed [21:0] NEG_B      = 22'sd1854524;           // round(1.769 * 2^20), fits 22 bits
            localparam signed [19:0] A_Q        = -20'sd302842;           // round(-0.2888 * 2^20), fits 20 bits
            localparam signed [31:0] MAXV       = (32'sd1 <<< 31) - 1;
            localparam signed [31:0] MINV       = -(32'sd1 <<< 31);

            localparam TW  = 23; // term width: NEG_B needs 22 bits + 1 margin
            localparam SQW = 23; // sq width: term^2 rescaled, bounded by NEG_B_real^2
            localparam PW  = 22; // poly width: bounded to [~0.096, 1.0] real
            localparam OPW = 23; // one_plus_l width: bounded to [0, 2.0] real

            // u = round(x * INV_SQRT2 / 2^FRAC), saturated to W bits (u spans
            // the same real domain as x, so it genuinely needs the full width).
            localparam signed [52:0] HALF_U = 53'sd1 <<< (FRAC - 1);
            localparam signed [52:0] MAXV_U = MAXV;
            localparam signed [52:0] MINV_U = MINV;
            wire signed [52:0] u_full = x * INV_SQRT2;
            wire signed [52:0] u_r    = (u_full + HALF_U) >>> FRAC;
            wire signed [31:0] u      = (u_r > MAXV_U) ? MAXV : (u_r < MINV_U) ? MINV : u_r[31:0];

            // mag_clipped is bounded by NEG_B (22 bits) by construction once
            // clipped -- term = mag_clipped - NEG_B is then provably in
            // [-NEG_B, 0].
            localparam signed [32:0] NEG_B_MAG = NEG_B;
            wire signed [32:0] mag_full    = u[31] ? (-{u[31], u}) : {u[31], u};
            wire signed [21:0] mag_clipped = (mag_full > NEG_B_MAG) ? NEG_B : mag_full[21:0];
            localparam signed [TW-1:0] NEG_B_T = NEG_B;
            wire signed [TW-1:0] mag_t = mag_clipped;
            wire signed [TW-1:0] term  = mag_t - NEG_B_T; // range [-NEG_B, 0]

            // sq = round(term^2/2^FRAC), provably in [0, round(1.769^2*2^20)]
            // (~3.28e6), needing 23 bits.
            localparam signed [2*TW-1:0] HALF_SQ = {{(2*TW-FRAC){1'b0}}, 1'b1} <<< (FRAC - 1);
            wire signed [2*TW-1:0] sq_full = term * term;
            wire signed [2*TW-1:0] sq_r    = (sq_full + HALF_SQ) >>> FRAC;
            wire signed [SQW-1:0]  sq      = sq_r[SQW-1:0];

            // poly = round(A_Q*sq/2^FRAC) + ONE, provably in
            // [round(0.096*2^20), 2^20] = [~100663, 1048576], needing 22 bits.
            localparam signed [19+SQW:0] HALF_PO = {{(20+SQW-FRAC){1'b0}}, 1'b1} <<< (FRAC - 1);
            localparam signed [PW-1:0]   ONE_P   = ONE;
            wire signed [19+SQW:0] poly_full = A_Q * sq;
            wire signed [19+SQW:0] poly_r    = (poly_full + HALF_PO) >>> FRAC;
            wire signed [PW-1:0]   poly      = poly_r[PW-1:0] + ONE_P;

            // l_term = (x==0) ? 0 : sgn(x)*poly, one_plus_l = ONE+l_term,
            // provably in [0, 2*2^20], needing 23 bits.
            localparam signed [OPW-1:0] ONE_OP = ONE;
            wire signed [PW-1:0]  l_term     = (x == 0) ? {PW{1'b0}} : (x[31] ? -poly : poly);
            wire signed [OPW-1:0] l_term_ext = l_term;
            wire signed [OPW-1:0] one_plus_l = ONE_OP + l_term_ext;

            // y = round(x * one_plus_l / 2^(FRAC+1)), saturated to W bits.
            localparam signed [W+OPW-1:0] HALF_Y = {{(W+OPW-FRAC-1){1'b0}}, 1'b1} <<< FRAC;
            localparam signed [W+OPW-1:0] MAXV_Y = MAXV;
            localparam signed [W+OPW-1:0] MINV_Y = MINV;
            wire signed [W+OPW-1:0] y_full = x * one_plus_l;
            wire signed [W+OPW-1:0] y_r    = (y_full + HALF_Y) >>> (FRAC + 1);
            assign y = (y_r > MAXV_Y) ? MAXV : (y_r < MINV_Y) ? MINV : y_r[31:0];
        end else if (W == 16 && FRAC == 8) begin : q8_8
            localparam signed [15:0] ONE        = 16'sd256;               // 1 * 2^8
            localparam signed [9:0]  INV_SQRT2  = 10'sd181;               // round(0.70710678 * 2^8), fits 10 bits
            localparam signed [9:0]  NEG_B      = 10'sd453;               // round(1.769 * 2^8), fits 10 bits
            localparam signed [8:0]  A_Q        = -9'sd74;                // round(-0.2888 * 2^8), fits 9 bits
            localparam signed [15:0] MAXV       = (16'sd1 <<< 15) - 1;
            localparam signed [15:0] MINV       = -(16'sd1 <<< 15);

            localparam TW  = 11;
            localparam SQW = 11;
            localparam PW  = 11;
            localparam OPW = 12;

            localparam signed [26:0] HALF_U = 27'sd1 <<< (FRAC - 1);
            localparam signed [26:0] MAXV_U = MAXV;
            localparam signed [26:0] MINV_U = MINV;
            wire signed [26:0] u_full = x * INV_SQRT2;
            wire signed [26:0] u_r    = (u_full + HALF_U) >>> FRAC;
            wire signed [15:0] u      = (u_r > MAXV_U) ? MAXV : (u_r < MINV_U) ? MINV : u_r[15:0];

            localparam signed [16:0] NEG_B_MAG = NEG_B;
            wire signed [16:0] mag_full   = u[15] ? (-{u[15], u}) : {u[15], u};
            wire signed [9:0]  mag_clipped = (mag_full > NEG_B_MAG) ? NEG_B : mag_full[9:0];
            localparam signed [TW-1:0] NEG_B_T = NEG_B;
            wire signed [TW-1:0] mag_t = mag_clipped;
            wire signed [TW-1:0] term  = mag_t - NEG_B_T;

            localparam signed [2*TW-1:0] HALF_SQ = {{(2*TW-FRAC){1'b0}}, 1'b1} <<< (FRAC - 1);
            wire signed [2*TW-1:0] sq_full = term * term;
            wire signed [2*TW-1:0] sq_r    = (sq_full + HALF_SQ) >>> FRAC;
            wire signed [SQW-1:0]  sq      = sq_r[SQW-1:0];

            localparam signed [8+SQW:0] HALF_PO = {{(9+SQW-FRAC){1'b0}}, 1'b1} <<< (FRAC - 1);
            localparam signed [PW-1:0]  ONE_P   = ONE;
            wire signed [8+SQW:0] poly_full = A_Q * sq;
            wire signed [8+SQW:0] poly_r    = (poly_full + HALF_PO) >>> FRAC;
            wire signed [PW-1:0]  poly      = poly_r[PW-1:0] + ONE_P;

            localparam signed [OPW-1:0] ONE_OP = ONE;
            wire signed [PW-1:0]  l_term     = (x == 0) ? {PW{1'b0}} : (x[15] ? -poly : poly);
            wire signed [OPW-1:0] l_term_ext = l_term;
            wire signed [OPW-1:0] one_plus_l = ONE_OP + l_term_ext;

            localparam signed [W+OPW-1:0] HALF_Y = {{(W+OPW-FRAC-1){1'b0}}, 1'b1} <<< FRAC;
            localparam signed [W+OPW-1:0] MAXV_Y = MAXV;
            localparam signed [W+OPW-1:0] MINV_Y = MINV;
            wire signed [W+OPW-1:0] y_full = x * one_plus_l;
            wire signed [W+OPW-1:0] y_r    = (y_full + HALF_Y) >>> (FRAC + 1);
            assign y = (y_r > MAXV_Y) ? MAXV : (y_r < MINV_Y) ? MINV : y_r[15:0];
        end
    endgenerate
endmodule
