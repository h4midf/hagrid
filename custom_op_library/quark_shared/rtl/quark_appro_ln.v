// QUARK (ICCAD 2025) Eq.6-9: approx ln(x) for x > 0. log2(x) = qM + poly, qM the
// integer exponent (leading-one detection), poly a quadratic fit to log2 of the
// mantissa qN in [1,2). Then ln(x) = log2(x) * ln2, ln2 ~= 0.6875 = 1 - 1/4 - 1/16
// (Eq.6). Max abs err ~0.06 over x in [0.001, 100]; matches appro_ln() in
// run_scripts/measure_accuracy/quark_bert_large.py.
//
// qM is added before the ln2 shift-subtract, as Eq.8's log2(x) = qM + log2(qN)
// requires (the paper's pseudocode uses one name for both the Eq.9 polynomial and the
// final result).
//
// Widening uses `wire signed [N:0] WIDE = NARROW;`, never `{...}` in a signed
// comparison: a concatenation is unsigned.
module quark_appro_ln #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,   // x > 0 required
    output wire signed [W-1:0] y    // approx ln(x)
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [2*W-1:0] MAXV2 = MAXV;
    localparam signed [2*W-1:0] MINV2 = MINV;
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);

    function [$clog2(W)-1:0] msb_pos;
        input [W-1:0] v;
        integer k;
        reg found;
        begin
            msb_pos = 0;
            found = 1'b0;
            for (k = W - 1; k >= 0; k = k - 1) begin
                if (!found && v[k]) begin
                    msb_pos = k[$clog2(W)-1:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [W-1:0] xu = x[W-1:0]; // x > 0 by contract, safe to treat as unsigned
    wire [$clog2(W)-1:0] p = msb_pos(xu);
    wire signed [W-1:0]  p_s = {{(W-$clog2(W)){1'b0}}, p}; // value-safe: p always >= 0
    wire signed [W-1:0]  qM  = p_s - FRAC; // plain integer exponent

    // mantissa fractional bits (shift left or right to FRAC bits)
    wire [W-1:0] frac_mask = ({W{1'b1}}) >> (W - FRAC); // FRAC-bit all-ones mask, zero above
    wire [W-1:0] frac_bits = (p >= FRAC) ? ((xu >> (p - FRAC)) & frac_mask)
                                          : ((xu << (FRAC - p)) & frac_mask);
    // frac_bits is already < 2^FRAC (< 2^(W-1)) after masking, so adding
    // it directly to ONE is safe without any further slicing.
    wire signed [W-1:0] qN = (1 <<< FRAC) + $signed(frac_bits); // mantissa in [1,2), Q.FRAC

    localparam signed [W-1:0] C0 = (W == 32 && FRAC == 20) ? -32'sd353290 : -16'sd86;  // round(-0.3369*2^FRAC)
    localparam signed [W-1:0] C1 = (W == 32 && FRAC == 20) ? 32'sd2091663 : 16'sd511;   // round(1.995*2^FRAC)
    localparam signed [W-1:0] C2 = (W == 32 && FRAC == 20) ? -32'sd1730150 : -16'sd422; // round(-1.65*2^FRAC)

    wire signed [2*W-1:0] qN2_full = $signed(qN) * $signed(qN);
    wire signed [2*W-1:0] qN2_r    = (qN2_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] c0qn2_full = $signed(C0) * qN2_r;
    wire signed [2*W-1:0] c0qn2_r    = (c0qn2_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] c1qn_full  = $signed(C1) * $signed(qN);
    wire signed [2*W-1:0] c1qn_r     = (c1qn_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] poly_wide  = c0qn2_r + c1qn_r + $signed(C2);
    wire signed [W-1:0]   poly = (poly_wide > MAXV2) ? MAXV[W-1:0] :
                                  (poly_wide < MINV2) ? MINV[W-1:0] : poly_wide[W-1:0];

    wire signed [W+7:0] qM_shifted = {{8{qM[W-1]}}, qM} <<< FRAC; // value-safe concat: feeds arithmetic only
    // A part-select (qM_shifted[W-1:0]) is ALWAYS unsigned in Verilog,
    // even though qM_shifted itself is declared signed -- used directly
    // inside a signed `+` this silently corrupted the sum (confirmed
    // directly: x=1 gave log2x_full as a huge unsigned-reinterpreted
    // value instead of -20963297). Routed through a named signed wire
    // first, which is the safe, already-proven pattern.
    wire signed [W-1:0] qM_shifted_narrow = qM_shifted[W-1:0];
    wire signed [W:0]   log2x_full = qM_shifted_narrow + poly;
    wire signed [W-1:0] log2x = (log2x_full > MAXV) ? MAXV[W-1:0] :
                                 (log2x_full < MINV) ? MINV[W-1:0] : log2x_full[W-1:0];

    // y = log2x - log2x/4 - log2x/16 (*0.6875 ~= ln2)
    assign y = log2x - (log2x >>> 2) - (log2x >>> 4);
endmodule
