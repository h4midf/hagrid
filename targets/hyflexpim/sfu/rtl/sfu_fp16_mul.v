// FP16 multiply, round to nearest even. Subnormal inputs read as zero, underflow
// flushes to +0, overflow saturates to +-65504, no Inf / NaN (fp16_ref.py).
// Stages (registered after each; 490 ps of logic per stage at NanGate45):
// 0 mantissa product in carry-save form (sfu_csa_mul), 1 final add,
// 2 normalize + round + pack.
module sfu_fp16_mul #(
    parameter PIPE = 1
) (
    input  wire        clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] y
);
    // ---- stage 0: flush, exponent, product of the 11-bit mantissas in carry-save form
    wire [4:0]  ea = a[14:10];
    wire [4:0]  eb = b[14:10];
    wire        zero = (ea == 5'd0) | (eb == 5'd0);
    wire signed [7:0] e = $signed({3'b000, ea}) + $signed({3'b000, eb}) - 8'sd15;
    wire [21:0] ps, pc;
    sfu_csa_mul #(.A(11), .B(11)) csa (
        .a({1'b1, a[9:0]}), .b({1'b1, b[9:0]}), .addend(22'd0), .sum(ps), .carry(pc));

    wire [21:0] ps1, pc1;
    wire signed [7:0] e1;
    wire        s1, z1;
    sfu_pipe #(.W(22 + 22 + 8 + 2), .EN(PIPE)) p0 (
        .clk(clk), .d({ps, pc, e, a[15] ^ b[15], zero}), .q({ps1, pc1, e1, s1, z1}));

    // ---- stage 1: product = sum + carry (22 bits; the product is below 2^22)
    wire [22:0] prod_raw;
    sfu_prefix_add #(.W(22)) add1 (.a(ps1), .b(pc1), .cin(1'b0), .s(prod_raw));
    wire [21:0] prod = prod_raw[21:0];

    wire [21:0] prod2;
    wire signed [7:0] e2;
    wire        s2, z2;
    sfu_pipe #(.W(22 + 8 + 2), .EN(PIPE)) p1 (
        .clk(clk), .d({prod, e1, s1, z1}), .q({prod2, e2, s2, z2}));

    // ---- stage 2: normalize (product in [1, 4)), round to nearest even, pack
    wire        top = prod2[21];
    wire [10:0] m   = top ? prod2[21:11] : prod2[20:10];
    wire        g   = top ? prod2[10] : prod2[9];
    wire        st  = top ? (prod2[9:0] != 10'd0) : (prod2[8:0] != 9'd0);
    wire        up  = g & (st | m[0]);
    wire [11:0] mr;                          // m + up, as a prefix increment
    sfu_prefix_add #(.W(11)) inc (.a(m), .b(11'd0), .cin(up), .s(mr));
    // exponent + top + rounding carry, from candidates computed in parallel
    wire signed [7:0] e_p1 = e2 + 8'sd1;
    wire signed [7:0] e_p2 = e2 + 8'sd2;
    wire signed [7:0] ef = (top & mr[11]) ? e_p2 : (top | mr[11]) ? e_p1 : e2;
    wire [9:0]  frac = mr[11] ? 10'd0 : mr[9:0];
    assign y = z2             ? 16'h0000 :
               (ef >= 8'sd31) ? {s2, 15'h7BFF} :
               (ef <= 8'sd0)  ? 16'h0000 :
                                {s2, ef[4:0], frac};
endmodule
