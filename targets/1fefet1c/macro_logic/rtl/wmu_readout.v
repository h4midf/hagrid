// WMU readout of one 1F1C array: turns a row read into stored values and puts one
// value at a time on the tile bus.
//   A WMU read is two ADC conversions per column: the bitline reference (code1), then
//   the cell shared onto it (code2) (spice/mac/cap_rw.py). While hold is high, a latch
//   per column follows code1; it keeps it once hold falls. The stored 2-bit level is
//   code2 - code1 compared against the thresholds T1 < T2 < T3.
//   CELLS_PER_VALUE columns of 2 bits make one VALUE_BITS value, so a row read gives
//   COLS / CELLS_PER_VALUE values; sel picks the one on the output. Only the selected
//   value's columns are decoded: CELLS_PER_VALUE subtract-and-compare units, shared
//   over the values.
module wmu_readout #(
    parameter COLS = 128,
    parameter CODE = 7,
    parameter CELLS_PER_VALUE = 16,
    parameter T1 = 2,
    parameter T2 = 6,
    parameter T3 = 11
) (
    input  wire                                   hold,
    input  wire [COLS*CODE-1:0]                   codes,
    input  wire [$clog2(COLS/CELLS_PER_VALUE)-1:0] sel,
    output wire [2*CELLS_PER_VALUE-1:0]           value
);
    localparam VB = CELLS_PER_VALUE * CODE;

    reg [COLS*CODE-1:0] ref_codes;
    always @* begin
        if (hold)
            ref_codes = codes;
    end

    wire [VB-1:0] code2 = codes[sel * VB +: VB];
    wire [VB-1:0] code1 = ref_codes[sel * VB +: VB];

    genvar i;
    generate
        for (i = 0; i < CELLS_PER_VALUE; i = i + 1) begin : dec
            wire signed [CODE:0] diff = $signed({1'b0, code2[i * CODE +: CODE]}) - $signed({1'b0, code1[i * CODE +: CODE]});
            assign value[2 * i +: 2] = (diff >= T3) ? 2'd3 : (diff >= T2) ? 2'd2 : (diff >= T1) ? 2'd1 : 2'd0;
        end
    endgenerate
endmodule
