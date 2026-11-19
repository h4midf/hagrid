// FP16 x 2^n for an int8 n: only the exponent changes; underflow flushes to +0,
// overflow saturates to +-65504, subnormal x reads as zero (fp16_ref.ldexp).
// Used by exp (sfu_lane): e^x = 2^n e^r.
// One stage of logic (the lane registers the output).
module sfu_fp16_ldexp (
    input  wire [15:0] a,
    input  wire [7:0]  n,
    output wire [15:0] y
);
    wire [4:0]        e  = a[14:10];
    wire signed [9:0] en = $signed({5'd0, e}) + $signed({{2{n[7]}}, n});
    assign y = (e == 5'd0)    ? 16'h0000 :
               (en >= 10'sd31) ? {a[15], 15'h7BFF} :
               (en <= 10'sd0)  ? 16'h0000 :
                                 {a[15], en[4:0], a[9:0]};
endmodule
