// Stimulus for test_rram_shift_add.py: 6 input bits over 8 columns, one code per cycle;
// the code of column c at bit b is (c * 37 + b * 11 + SEED) mod 8.
module rram_shift_add_tb #(parameter SEED = 5) (input wire clk, output wire [71:0] acc_flat);
    reg [5:0] t = 0;
    wire [2:0] c = t[2:0];
    wire [2:0] b = t[5:3];
    wire running = (t < 48);
    wire [15:0] raw = c * 37 + b * 11 + SEED;

    reg [8:0] acc [0:7];
    integer i;
    initial for (i = 0; i < 8; i = i + 1) acc[i] = 0;

    wire [8:0] acc_out;
    rram_shift_add dut (.first(b == 3'd0), .code(raw[2:0]), .acc_in(acc[c]), .acc_out(acc_out));

    always @(posedge clk)
        if (running) begin
            acc[c] <= acc_out;
            t <= t + 6'd1;
        end

    assign acc_flat = {acc[7], acc[6], acc[5], acc[4], acc[3], acc[2], acc[1], acc[0]};
endmodule
