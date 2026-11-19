// Test harness for wmu_readout (simulation only, never synthesized). Cycle 0 (hold):
// column j shows its reference code R_j = (j * 29 + 7) mod 100. Cycle 1: it shows
// R_j + D[(j + SEED) mod 4], D = {0, DIFF1, DIFF2, DIFF3} (the read differences of the
// four stored levels). sel = SEL throughout.
module wmu_readout_tb #(
    parameter COLS = 128, parameter CODE = 7, parameter CELLS_PER_VALUE = 16,
    parameter T1 = 2, parameter T2 = 6, parameter T3 = 11,
    parameter DIFF1 = 4, parameter DIFF2 = 8, parameter DIFF3 = 13,
    parameter SEED = 0, parameter SEL = 0
) (
    input  wire                         clk,
    output wire [2*CELLS_PER_VALUE-1:0] value
);
    localparam SW = $clog2(COLS / CELLS_PER_VALUE);
    localparam [SW-1:0] SEL_W = SEL;

    reg [1:0] cnt = 2'd0;
    always @(posedge clk) cnt <= cnt + 2'd1;

    wire [COLS*CODE-1:0] codes;
    genvar j;
    generate
        for (j = 0; j < COLS; j = j + 1) begin : col
            localparam integer R = (j * 29 + 7) % 100;
            localparam integer L = (j + SEED) % 4;
            localparam integer D = (L == 0) ? 0 : (L == 1) ? DIFF1 : (L == 2) ? DIFF2 : DIFF3;
            assign codes[j * CODE +: CODE] = (cnt == 2'd0) ? R : R + D;
        end
    endgenerate

    wmu_readout #(.COLS(COLS), .CODE(CODE), .CELLS_PER_VALUE(CELLS_PER_VALUE), .T1(T1), .T2(T2), .T3(T3)) dut (
        .hold(cnt == 2'd0), .codes(codes), .sel(SEL_W), .value(value)
    );
endmodule
