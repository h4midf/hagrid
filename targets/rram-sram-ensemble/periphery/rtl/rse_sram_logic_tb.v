// Stimulus for test_rse_sram_logic.py: loads the 64 input bits, then runs one macro pass.
// Input bit i is bit 1 of (i * 5 + SEED); SRAM bit p at address a is bit 2 of
// (a * 13 + p * 7 + SEED).
module rse_sram_logic_tb #(parameter SEED = 5) (input wire clk, output wire [143:0] out_flat);
    reg [9:0] t = 0;
    wire rst = (t < 2);
    wire in_we = (t >= 2) && (t < 66);
    wire [5:0] in_addr = t - 10'd2;
    wire [15:0] in_raw = in_addr * 5 + SEED;
    wire start = (t == 67);

    wire [6:0] addr;
    wire ren, done;
    wire [15:0] rdata;
    genvar p;
    generate
        for (p = 0; p < 16; p = p + 1) begin : mem
            wire [15:0] raw = addr * 13 + p * 7 + SEED;
            assign rdata[p] = raw[2];
        end
    endgenerate

    rse_sram_logic dut (.clk(clk), .rst(rst), .start(start), .in_we(in_we), .in_addr(in_addr), .in_bit(in_raw[1]),
                        .sram_rdata(rdata), .sram_addr(addr), .sram_ren(ren), .done(done), .out_flat(out_flat));

    always @(posedge clk) t <= t + 10'd1;
endmodule
