// QUARK's GELU (Eq.11-14): for |x| < 2.4, GELU(x) ~= x*sigma(1.702x), and
// sigma(1.702x) is the first output of a 2-element softmax of [0, -1.702x] (Eq.12).
// So it instantiates quark_softmax_row8 with [0, -1.702x] on lanes 0/1 and a very
// negative sentinel on lanes 2-7, whose exp is 0. For |x| >= 2.4, y = ReLU(x).
//
// No outer 1.702 factor: the paper's printed Eq.13 has one, a typo (with it
// GELU(1.0) = 1.4486, without 0.8511, true 0.8413).
module quark_gelu #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));
    localparam signed [W-1:0] THRESH = (W == 32 && FRAC == 20) ? 32'sd2516582 : 16'sd614; // round(2.4*2^FRAC)
    localparam signed [W-1:0] C_1702 = (W == 32 && FRAC == 20) ? 32'sd1784739 : 16'sd436;  // round(1.702*2^FRAC)
    // The sentinel must not overflow appro_exp's xshift = x*1.4375 at W bits: -100
    // overflows at W=16 (|x| < ~22797 Q8.8 units), -8000 Q8.8 (~-31, exp ~1e-14) does not.
    localparam signed [W-1:0] SENTINEL = (W == 32 && FRAC == 20) ? -32'sd1048576000 : -16'sd8000;

    wire signed [2*W-1:0] m_full = $signed(x) * $signed(C_1702);
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    wire signed [2*W-1:0] m_r = (m_full + HALF) >>> FRAC;
    localparam signed [2*W-1:0] MAXV2 = MAXV;
    localparam signed [2*W-1:0] MINV2 = MINV;
    wire signed [W-1:0] s1 = (m_r > MAXV2) ? MAXV[W-1:0] : (m_r < MINV2) ? MINV[W-1:0] : -m_r[W-1:0]; // -1.702*x

    wire signed [W-1:0] p0, p1, p_unused2, p_unused3, p_unused4, p_unused5, p_unused6, p_unused7;
    quark_softmax_row8 #(.W(W), .FRAC(FRAC)) u_sm (
        .s0({W{1'b0}}), .s1(s1),
        .s2(SENTINEL), .s3(SENTINEL), .s4(SENTINEL), .s5(SENTINEL), .s6(SENTINEL), .s7(SENTINEL),
        .p0(p0), .p1(p1),
        .p2(p_unused2), .p3(p_unused3), .p4(p_unused4), .p5(p_unused5), .p6(p_unused6), .p7(p_unused7)
    );

    // inner = x * sigma(1.702x) = x * p0
    wire signed [2*W-1:0] inner_full = $signed(x) * $signed(p0);
    wire signed [2*W-1:0] inner_r = (inner_full + HALF) >>> FRAC;
    wire signed [W-1:0] inner = (inner_r > MAXV2) ? MAXV[W-1:0] : (inner_r < MINV2) ? MINV[W-1:0] : inner_r[W-1:0];

    wire signed [W-1:0] outer = x[W-1] ? {W{1'b0}} : x; // ReLU(x)

    wire signed [W-1:0] abs_x = x[W-1] ? -x : x;
    localparam signed [W-1:0] THRESH_C = THRESH;
    assign y = (abs_x < THRESH_C) ? inner : outer;
endmodule
