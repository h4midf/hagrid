// int8 -> FP16, exact (|n| <= 128 fits the 11-bit significand) (fp16_ref.int8_to_fp16).
// Used by exp (sfu_lane): the integer part n of x log2(e) back to FP16 for f = t - n.
// One stage of logic (the lane registers the output).
module sfu_int8_to_fp16 (
    input  wire [7:0]  n,
    output wire [15:0] y
);
    wire       s   = n[7];
    wire [7:0] mag = s ? (~n + 8'd1) : n;     // 128 for n = -128
    reg  [3:0] msb;                           // position of the leading one, 0..7
    integer i;
    always @(*) begin
        msb = 4'd0;
        for (i = 0; i < 8; i = i + 1)
            if (mag[i]) msb = i[3:0];
    end
    // mantissa: the bits below the leading one, left-aligned in 10 bits
    wire [17:0] aligned = {mag, 10'd0} >> msb;
    wire [4:0]  e = 5'd15 + {1'b0, msb};
    assign y = (mag == 8'd0) ? 16'h0000 : {s, e, aligned[9:0]};
endmodule
