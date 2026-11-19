// FP16 -> int8, round to nearest even, saturating to [-128, 127]; subnormals read as
// zero (fp16_ref.to_int8). HyFlexPIM Sec. 3.1: the SFU's results 'are converted to
// the integer format for further processing'.
// x = M * 2^(e - 25): |x| < 0.5 for e <= 13 (rounds to 0), |x| >= 256 for e >= 23
// (saturates); otherwise |x| = M >> (25 - e) with guard and sticky from the bits
// shifted out.
// One stage of logic (the lane registers the output).
module sfu_fp16_to_int8 (
    input  wire [15:0] a,
    output wire [7:0]  y
);
    wire [4:0]  e = a[14:10];
    wire        s = a[15];
    wire [10:0] m = {1'b1, a[9:0]};
    wire        small = (e <= 5'd13);
    wire        big   = (e >= 5'd23);
    wire [4:0]  k = 5'd25 - e;               // shift 3..11 for e = 14..22
    // {M, 11 zero bits} >> k: integer part in the top 11 bits, the rest below
    wire [21:0] sh = {m, 11'd0} >> k;
    wire [10:0] ip = sh[21:11];
    wire        g  = sh[10];
    wire        st = (sh[9:0] != 10'd0);
    wire        up = g & (st | ip[0]);
    wire [11:0] q  = {1'b0, ip} + {11'd0, up};  // |x| rounded, up to 256
    // saturate: +127 above 127, -128 below -128
    wire        over_pos = ~s & (big | (q > 12'd127));
    wire        over_neg =  s & (big | (q > 12'd128));
    wire [7:0]  mag = q[7:0];
    assign y = small    ? 8'd0 :
               over_pos ? 8'd127 :
               over_neg ? 8'h80 :
               s        ? (~mag + 8'd1) : mag;
endmodule
