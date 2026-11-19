// FP16 add / subtract (y = a + b, or a - b when sub = 1), round to nearest even.
// Subnormal inputs read as zero, underflow flushes to +0, overflow saturates to
// +-65504, no Inf / NaN (fp16_ref.py).
// Stages (registered after each; 1 GHz at 65 nm = 490 ps of logic per stage at
// NanGate45, see characterize_sfu.py): 0 order by magnitude, 1 align, 2 add,
// 3 normalize, 4 round + pack.
module sfu_fp16_add #(
    parameter PIPE = 1
) (
    input  wire        clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    input  wire        sub,
    output wire [15:0] y
);
    // ---- stage 1: flush, order by magnitude, align the smaller operand
    wire [4:0]  ea = a[14:10];
    wire [4:0]  eb = b[14:10];
    wire [14:0] mag_a = (ea == 5'd0) ? 15'd0 : a[14:0];
    wire [14:0] mag_b = (eb == 5'd0) ? 15'd0 : b[14:0];
    wire        sa = a[15];
    wire        sb = b[15] ^ sub;
    // mag_b > mag_a exactly when mag_a - mag_b borrows (no carry out of a + ~b + 1)
    wire [15:0] cmp;
    sfu_prefix_add #(.W(15)) cmp0 (.a(mag_a), .b(~mag_b), .cin(1'b1), .s(cmp));
    wire        swap = ~cmp[15];
    // both exponent differences and both mantissas in parallel with the compare
    wire [4:0]  fe_a = mag_a[14:10];
    wire [4:0]  fe_b = mag_b[14:10];
    wire [4:0]  d_ab = fe_a - fe_b;
    wire [4:0]  d_ba = fe_b - fe_a;
    wire [10:0] fm_a = (fe_a == 5'd0) ? 11'd0 : {1'b1, mag_a[9:0]};
    wire [10:0] fm_b = (fe_b == 5'd0) ? 11'd0 : {1'b1, mag_b[9:0]};
    wire        s_l = swap ? sb : sa;
    wire        s_s = swap ? sa : sb;
    wire [4:0]  e_l = swap ? fe_b : fe_a;
    wire [10:0] m_l = swap ? fm_b : fm_a;
    wire [10:0] m_s = swap ? fm_a : fm_b;
    wire [4:0]  d   = swap ? d_ba : d_ab;

    wire [4:0]  e0, d0;
    wire [10:0] ml0, ms0;
    wire        sl0, ss0;
    sfu_pipe #(.W(5 + 5 + 11 + 11 + 2), .EN(PIPE)) p0 (
        .clk(clk), .d({e_l, d, m_l, m_s, s_l, s_s}), .q({e0, d0, ml0, ms0, sl0, ss0}));

    // the smaller mantissa with 3 extra bits (guard, round, sticky), shifted right by d
    // sticky: the mantissa bits shifted below the 3 extra bits (the low d - 3 bits),
    // from a mask decoded from d, in parallel with the shift
    wire [13:0] ext_s = {ms0, 3'b000};
    wire [13:0] shifted = ext_s >> d0;
    wire [10:0] lowmask = (d0 <= 5'd3)  ? 11'd0 :
                          (d0 >= 5'd14) ? 11'h7FF : ((11'd1 << (d0 - 5'd3)) - 11'd1);
    wire        sticky1 = (ms0 & lowmask) != 11'd0;
    wire [13:0] al = {shifted[13:1], shifted[0] | sticky1};

    wire [4:0]  e1;
    wire [13:0] ml1, al1;
    wire        s1, eff_sub1;
    sfu_pipe #(.W(5 + 14 + 14 + 2), .EN(PIPE)) p1 (
        .clk(clk), .d({e0, {ml0, 3'b000}, al, sl0, sl0 ^ ss0}),
        .q({e1, ml1, al1, s1, eff_sub1}));

    // ---- stage 2: add or subtract (|large| >= |small|, so no negative result)
    // subtract as ml + ~al + 1; the carry out of a subtraction is dropped
    wire [14:0] sum_raw;
    sfu_prefix_add #(.W(14)) add2 (
        .a(ml1), .b(eff_sub1 ? ~al1 : al1), .cin(eff_sub1), .s(sum_raw));
    wire [14:0] sum = {sum_raw[14] & ~eff_sub1, sum_raw[13:0]};

    wire [4:0]  e2;
    wire [14:0] sum2;
    wire        s2;
    sfu_pipe #(.W(5 + 15 + 1), .EN(PIPE)) p2 (.clk(clk), .d({e1, sum, s1}), .q({e2, sum2, s2}));

    // ---- stage 3: count leading zeros, normalize so the leading one is at bit 13
    reg  [3:0] lz;
    integer i;
    always @(*) begin
        lz = 4'd14;
        for (i = 0; i < 14; i = i + 1)
            if (sum2[i]) lz = 4'd13 - i[3:0];
    end
    wire        carry = sum2[14];
    wire [13:0] norm  = carry ? {sum2[14:2], sum2[1] | sum2[0]} : (sum2[13:0] << lz);
    // biased exponent, signed: e2 + 1 on carry, e2 - lz otherwise
    wire signed [7:0] en = carry ? $signed({3'b000, e2}) + 8'sd1
                                 : $signed({3'b000, e2}) - $signed({4'b0000, lz});
    wire        zero = (sum2 == 15'd0);

    wire [13:0] norm3;
    wire signed [7:0] en3;
    wire        s3, zero3;
    sfu_pipe #(.W(14 + 8 + 2), .EN(PIPE)) p3 (
        .clk(clk), .d({norm, en, s2, zero}), .q({norm3, en3, s3, zero3}));

    // ---- stage 4: round to nearest even, flush / saturate, pack
    wire        up  = norm3[2] & (norm3[1] | norm3[0] | norm3[3]);
    wire [11:0] mr  = {1'b0, norm3[13:3]} + {11'd0, up};
    wire [9:0]  frac = mr[11] ? 10'd0 : mr[9:0];
    wire signed [7:0] ef = mr[11] ? en3 + 8'sd1 : en3;
    assign y = zero3          ? 16'h0000 :
               (ef >= 8'sd31) ? {s3, 15'h7BFF} :
               (ef <= 8'sd0)  ? 16'h0000 :
                                {s3, ef[4:0], frac};
endmodule
