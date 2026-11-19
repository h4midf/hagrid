// QUARK (ICCAD 2025) Eq.3-5: approx exp(x) for x <= 0, base-2 shift-add range
// reduction (x + x/2 - x/16 ~= x/ln2, as SOLE's log2exp) and a quadratic fit. Unlike
// log2exp, qI = ceil(xshift) (<= 0) and full precision is kept. Max abs err ~0.002
// (approximation_error.py); matches appro_exp() in
// run_scripts/measure_accuracy/quark_bert_large.py.
//
// Widening uses `wire signed [N:0] WIDE = NARROW;`, never `{...}` in a signed
// comparison: a concatenation is unsigned.
// x must not be so negative that x + x/2 - x/16 overflows W bits (x = MINV wraps).
module quark_appro_exp #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,   // x <= 0 required (see above)
    output wire signed [W-1:0] y    // approx exp(x)
);
    localparam signed [W-1:0] ONE = 1 <<< FRAC;
    localparam signed [W:0]   MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0]   MINV = -(1 <<< (W - 1));

    // xshift = x + x/2 - x/16 (~= x/ln2), always <= 0 since x<=0.
    wire signed [W-1:0] xshift = x + (x >>> 1) - (x >>> 4);

    // qI = ceil(xshift_real): floor via arithmetic shift, +1 if any
    // fractional bits are set (xshift is not an exact multiple of 2^FRAC).
    wire signed [W-1:0] floor_part = xshift >>> FRAC;
    wire [FRAC-1:0]     low_bits   = xshift[FRAC-1:0];
    wire                has_frac   = |low_bits;
    wire signed [W-1:0] qI = floor_part + (has_frac ? {{(W-1){1'b0}}, 1'b1} : {W{1'b0}});

    // qF = xshift - (qI << FRAC), in (-ONE, 0].
    wire signed [W+7:0] qI_shifted = {{8{qI[W-1]}}, qI} <<< FRAC; // concat is fine: feeds arithmetic only
    wire signed [W-1:0] qF = xshift - qI_shifted[W-1:0];

    // xa = 0.1713*qF^2 + 0.6674*qF + 0.998, Q.FRAC coefficients.
    localparam signed [W-1:0] C0 = (W == 32 && FRAC == 20) ? 32'sd179625 : 16'sd44;  // round(0.1713*2^FRAC)
    localparam signed [W-1:0] C1 = (W == 32 && FRAC == 20) ? 32'sd699896 : 16'sd171; // round(0.6674*2^FRAC)
    localparam signed [W-1:0] C2 = (W == 32 && FRAC == 20) ? 32'sd1046478 : 16'sd256; // round(0.998*2^FRAC)
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);

    wire signed [2*W-1:0] qF2_full = $signed(qF) * $signed(qF);
    wire signed [2*W-1:0] qF2_r    = (qF2_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] c0qf2_full = $signed(C0) * qF2_r;
    wire signed [2*W-1:0] c0qf2_r    = (c0qf2_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] c1qf_full  = $signed(C1) * $signed(qF);
    wire signed [2*W-1:0] c1qf_r     = (c1qf_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] xa_wide    = c0qf2_r + c1qf_r + $signed(C2);
    localparam signed [2*W-1:0] MAXV2 = MAXV;
    localparam signed [2*W-1:0] MINV2 = MINV;
    wire signed [W-1:0]   xa = (xa_wide > MAXV2) ? MAXV[W-1:0] :
                                (xa_wide < MINV2) ? MINV[W-1:0] : xa_wide[W-1:0];

    // y = xa * 2^qI = xa >>> (-qI) (qI<=0 always, so -qI>=0).
    wire signed [W-1:0] shift_amt = -qI;
    assign y = (shift_amt >= W) ? {W{1'b0}} : (xa >>> shift_amt[$clog2(W)-1:0]);
endmodule
