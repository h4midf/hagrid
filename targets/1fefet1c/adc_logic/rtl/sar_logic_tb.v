// Test harness for sar_logic (simulation only, never synthesized): one conversion of
// the input level CODE with an ideal comparator. samp is high in the first of every
// 8 cycles (one sample cycle, then 7 trial cycles, as in sar_core_tb.sp). The DAC
// level is enref read as a binary number (bit k = 2^k unit capacitors to Vref); with
// the input at (CODE + 0.5) LSB, the comparator keeps a trial bit when that level is
// at most CODE. After the 8th rising edge, b must equal CODE.
module sar_logic_tb #(parameter CODE = 0) (
    input  wire       clk,
    output wire [6:0] b
);
    reg [2:0] cnt = 3'd0;
    always @(posedge clk) cnt <= cnt + 3'd1;

    wire       samp = (cnt == 3'd0);
    wire [6:0] enref, engnd;
    wire       clkc, sampb;
    wire       cmp = (enref <= CODE);

    sar_logic dut (
        .clk(clk), .samp(samp), .cmp(cmp),
        .b(b), .enref(enref), .engnd(engnd), .clkc(clkc), .sampb(sampb)
    );
endmodule
