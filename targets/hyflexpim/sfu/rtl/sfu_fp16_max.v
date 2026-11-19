// FP16 max: the larger of a and b, subnormals (and -0) read as +0, ties give a
// (fp16_ref.fmax). One step of max search; the search over a row feeds the result
// back as an operand.
// One stage of logic (the lane registers the output).
module sfu_fp16_max (
    input  wire [15:0] a,
    input  wire [15:0] b,
    output wire [15:0] y
);
    wire [15:0] fa = (a[14:10] == 5'd0) ? 16'h0000 : a;
    wire [15:0] fb = (b[14:10] == 5'd0) ? 16'h0000 : b;
    // |b| > |a| exactly when |a| - |b| borrows
    wire [15:0] cmp;
    sfu_prefix_add #(.W(15)) cmp0 (.a(fa[14:0]), .b(~fb[14:0]), .cin(1'b1), .s(cmp));
    wire        b_mag_gt = ~cmp[15];
    wire        a_mag_gt = cmp[15] & (cmp[14:0] != 15'd0);
    // b > a: b positive and a negative, or both positive and |b| > |a|, or both
    // negative and |b| < |a|
    wire        b_gt = (~fb[15] & fa[15]) | (~fb[15] & ~fa[15] & b_mag_gt) | (fb[15] & fa[15] & a_mag_gt);
    assign y = b_gt ? fb : fa;
endmodule
