// FP16 square root, round to nearest even. Negative inputs, zero and subnormals give
// +0 (fp16_ref.py). The result never overflows or underflows.
// x = M * 2^(e - 25) with M the 11-bit mantissa. With the exponent made even, the
// radicand is N = M * 2^12 (e odd) or M * 2^13 (e even), a 24-bit integer whose
// integer square root is the 12-bit root (integer bit, 10 fraction bits, guard);
// the remainder gives sticky. Result exponent: (e + 15) / 2, rounded down.
// Restoring square root, one root bit per stage: R = 4 R + next two radicand bits;
// if R >= 4 S + 1 then R -= 4 S + 1, S = 2 S + 1, else S = 2 S.
// Stages (registered after each; 490 ps of logic per stage at NanGate45):
// 0 decode, 1..12 one root bit each, 13 round + pack.
module sfu_fp16_sqrt #(
    parameter PIPE = 1
) (
    input  wire        clk,
    input  wire [15:0] a,
    output wire [15:0] y
);
    localparam N = 12;                       // root bits
    localparam F = 5 + 1;                    // carried flags: result exponent, zero

    // ---- stage 0: decode; the radicand's nonzero bits N[23:12]
    wire [4:0]  e    = a[14:10];
    wire        zero = (e == 5'd0) | a[15];
    wire [10:0] m    = {1'b1, a[9:0]};
    wire [11:0] rad0 = e[0] ? {1'b0, m} : {m, 1'b0};
    wire [5:0]  esum = {1'b0, e} + 6'd15;

    wire [14:0]  r   [0:N];
    wire [N-1:0] s   [0:N];
    wire [11:0]  rad [0:N];
    wire [F-1:0] fl  [0:N];
    sfu_pipe #(.W(15 + N + 12 + F), .EN(PIPE)) p0 (
        .clk(clk), .d({15'd0, {N{1'b0}}, rad0, esum[5:1], zero}),
        .q({r[0], s[0], rad[0], fl[0]}));

    // ---- stages 1..N: one root bit each
    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : step
            wire [14:0] rx = {r[k][12:0], rad[k][11:10]};
            wire [14:0] t  = {s[k][N-2:0], 2'b01};
            wire [15:0] diff;                // rx - t; carry out set when rx >= t
            sfu_prefix_add #(.W(15)) sub_ (.a(rx), .b(~t), .cin(1'b1), .s(diff));
            wire        qk  = diff[15];
            wire [14:0] rem = qk ? diff[14:0] : rx;
            sfu_pipe #(.W(15 + N + 12 + F), .EN(PIPE)) p (
                .clk(clk), .d({rem, s[k][N-2:0], qk, rad[k][9:0], 2'b00, fl[k]}),
                .q({r[k + 1], s[k + 1], rad[k + 1], fl[k + 1]}));
        end
    endgenerate

    // ---- stage N + 1: round to nearest even, pack
    wire [N-1:0] sf = s[N];
    wire [4:0]  ef0 = fl[N][F-1:1];
    wire        zf  = fl[N][0];
    wire        sticky = (r[N] != 15'd0);
    wire        up  = sf[0] & (sticky | sf[1]);
    wire [11:0] mr  = {1'b0, sf[N-1:1]} + {11'd0, up};
    wire [4:0]  ef  = mr[11] ? ef0 + 5'd1 : ef0;
    wire [9:0]  frac = mr[11] ? 10'd0 : mr[9:0];
    assign y = zf ? 16'h0000 : {1'b0, ef, frac};
endmodule
