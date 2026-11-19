// Test harness for macro_ctrl (simulation only, never synthesized): start and load are
// high in cycle 0 only, with din = the byte PAT repeated.
module macro_ctrl_tb #(parameter ROWS = 128, parameter CONV_CYCLES = 8, parameter PAT = 8'hA5) (
    input  wire            clk,
    output wire [ROWS-1:0] wl,
    output wire            rst,
    output wire            mult,
    output wire            acc,
    output wire            samp,
    output wire            busy
);
    reg [4:0] cnt = 5'd0;
    always @(posedge clk) cnt <= cnt + 5'd1;

    wire [8*ROWS-1:0] pat = {ROWS{PAT[7:0]}};

    macro_ctrl #(.ROWS(ROWS), .CONV_CYCLES(CONV_CYCLES)) dut (
        .clk(clk), .start(cnt == 5'd0), .load(cnt == 5'd0), .din(pat[ROWS-1:0]),
        .wl(wl), .rst(rst), .mult(mult), .acc(acc), .samp(samp), .busy(busy)
    );
endmodule
