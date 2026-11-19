// Test harness for sfu_lane in exp mode (simulation only, never synthesized): rst in
// cycle 0, then exp j's input X[j] is applied from cycle 1 + 7 j (the lane samples it at
// phase 0) and its result is captured from ldexp_y at cycle 1 + 7 j + 71. X comes from
// XHEX (one FP16 value in hex per line, N lines).
module sfu_lane_exp_tb #(
    parameter N = 64,
    parameter XHEX = "exp_x.hex"
) (
    input  wire            clk,
    output wire [16*N-1:0] y_flat
);
    reg [15:0] xs [0:N-1];
    initial $readmemh(XHEX, xs);

    reg [15:0] t = 16'd0;
    always @(posedge clk) t <= t + 16'd1;
    wire [15:0] j_in = (t - 16'd1) / 16'd7;

    wire [15:0] ldexp_y;
    sfu_lane dut (
        .clk(clk), .rst(t == 16'd0), .exp_mode(1'b1),
        .exp_x(t != 16'd0 && j_in < N ? xs[j_in] : 16'h0000),
        .max_a(16'd0), .max_b(16'd0), .sub_a(16'd0), .sub_b(16'd0), .add_a(16'd0), .add_b(16'd0),
        .div_a(16'd0), .div_b(16'd0), .mul_a(16'd0), .mul_b(16'd0), .sqrt_a(16'd0), .int_a(16'd0),
        .i2f_n(8'd0), .ldexp_a(16'd0), .ldexp_n(8'd0),
        .max_y(), .sub_y(), .add_y(), .div_y(), .mul_y(), .sqrt_y(), .i2f_y(), .ldexp_y(ldexp_y),
        .int_y());

    reg [15:0] ys [0:N-1];
    wire [15:0] t_out = t - 16'd72;
    always @(posedge clk)
        if (t >= 16'd72 && t_out % 16'd7 == 16'd0 && t_out / 16'd7 < N)
            ys[t_out / 16'd7] <= ldexp_y;

    genvar k;
    generate
        for (k = 0; k < N; k = k + 1) begin : flat
            assign y_flat[16 * k +: 16] = ys[k];
        end
    endgenerate
endmodule
