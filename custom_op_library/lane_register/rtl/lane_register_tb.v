// Test harness for lane_register (simulation only, never synthesized). The data input
// is the byte PA repeated in cycle 0 and the byte PB repeated afterwards; load is high
// only in cycle LOAD_AT. After 3 rising edges, q must hold the data of cycle LOAD_AT.
module lane_register_tb #(parameter BITS = 96, parameter PA = 8'hA5, parameter PB = 8'h3C,
                          parameter LOAD_AT = 0) (
    input  wire            clk,
    output wire [BITS-1:0] q
);
    reg [1:0] cnt = 2'd0;
    always @(posedge clk) cnt <= cnt + 2'd1;

    wire [8*BITS-1:0] a = {BITS{PA[7:0]}};
    wire [8*BITS-1:0] b = {BITS{PB[7:0]}};
    wire [BITS-1:0]   d = (cnt == 2'd0) ? a[BITS-1:0] : b[BITS-1:0];

    lane_register #(.BITS(BITS)) dut (.clk(clk), .load(cnt == LOAD_AT), .d(d), .q(q));
endmodule
