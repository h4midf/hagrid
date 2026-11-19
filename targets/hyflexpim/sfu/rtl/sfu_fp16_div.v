// FP16 divide (y = a / b), round to nearest even. Subnormal inputs read as zero,
// underflow flushes to +0, overflow saturates to +-65504, no Inf / NaN; b = 0 gives
// +-65504 with the sign of a XOR b (+65504 for 0 / 0) (fp16_ref.py).
// Restoring division of the 11-bit mantissas, one quotient bit per stage: the
// dividend is put in [D, 2D) (exponent - 1 if ma < mb), so the 12 quotient bits are
// the integer bit, 10 fraction bits and the guard bit; the remainder gives sticky.
// Stages (registered after each; 490 ps of logic per stage at NanGate45):
// 0 compare + align, 1..12 one quotient bit each, 13 round + pack.
module sfu_fp16_div #(
    parameter PIPE = 1
) (
    input  wire        clk,
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] y
);
    localparam N = 12;                       // quotient bits
    localparam F = 8 + 3;                    // carried flags: exponent, sign, zero a, zero b

    // ---- stage 0: flush, compare mantissas, dividend in [D, 2D), exponent
    wire [4:0]  ea = a[14:10];
    wire [4:0]  eb = b[14:10];
    wire        za = (ea == 5'd0);
    wire        zb = (eb == 5'd0);
    wire [10:0] ma = {1'b1, a[9:0]};
    wire [10:0] mb = {1'b1, b[9:0]};
    wire [11:0] cmp;                         // ma - mb: borrow (no carry out) when ma < mb
    sfu_prefix_add #(.W(11)) cmp0 (.a(ma), .b(~mb), .cin(1'b1), .s(cmp));
    wire        lt = ~cmp[11];
    wire [11:0] x  = lt ? {ma, 1'b0} : {1'b0, ma};
    wire signed [7:0] e = $signed({3'b000, ea}) - $signed({3'b000, eb}) + 8'sd15
                          - (lt ? 8'sd1 : 8'sd0);

    // per stage: remainder (12), divisor (11), quotient so far (N), flags
    wire [11:0]    r   [0:N];
    wire [10:0]    d   [0:N];
    wire [N-1:0]   q   [0:N];
    wire [F-1:0]   fl  [0:N];
    sfu_pipe #(.W(12 + 11 + N + F), .EN(PIPE)) p0 (
        .clk(clk), .d({x, mb, {N{1'b0}}, e, a[15] ^ b[15], za, zb}),
        .q({r[0], d[0], q[0], fl[0]}));

    // ---- stages 1..N: q_k = (r >= d); r = 2 (q_k ? r - d : r)
    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : step
            wire [12:0] diff;                // r - d; carry out set when r >= d
            sfu_prefix_add #(.W(12)) sub_ (.a(r[k]), .b(~{1'b0, d[k]}), .cin(1'b1), .s(diff));
            wire        qk   = diff[12];
            wire [11:0] rem  = qk ? diff[11:0] : r[k];
            sfu_pipe #(.W(12 + 11 + N + F), .EN(PIPE)) p (
                .clk(clk), .d({rem[10:0], 1'b0, d[k], q[k][N-2:0], qk, fl[k]}),
                .q({r[k + 1], d[k + 1], q[k + 1], fl[k + 1]}));
        end
    endgenerate

    // ---- stage N + 1: round to nearest even, flush / saturate, pack
    wire [N-1:0] qf = q[N];
    wire signed [7:0] ef0 = fl[N][F-1:3];
    wire        sf  = fl[N][2];
    wire        zaf = fl[N][1];
    wire        zbf = fl[N][0];
    wire        sticky = (r[N] != 12'd0);
    wire        up  = qf[0] & (sticky | qf[1]);
    wire [11:0] mr  = {1'b0, qf[N-1:1]} + {11'd0, up};
    wire signed [7:0] ef = mr[11] ? ef0 + 8'sd1 : ef0;
    wire [9:0]  frac = mr[11] ? 10'd0 : mr[9:0];
    assign y = zbf            ? {sf & ~zaf, 15'h7BFF} :
               zaf            ? 16'h0000 :
               (ef >= 8'sd31) ? {sf, 15'h7BFF} :
               (ef <= 8'sd0)  ? 16'h0000 :
                                {sf, ef[4:0], frac};
endmodule
