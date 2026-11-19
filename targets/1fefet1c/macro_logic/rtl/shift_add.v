// Shift-and-add datapaths of one 1F1C array: turn the column ADC codes of each MAC
// (one input bit) into OUTS signed outputs, accumulated over the bit-serial input.
//   An output's weight spans CELLS cells of 2 bits (an int8 weight: 4 cells), so its
//   partial sum is code_0 + code_1 * 4 + code_2 * 16 + code_3 * 64 (PART bits).
//   Inputs arrive MSB first, two's complement: on the first (sign) bit acc = -partial,
//   then acc = 2 * acc + partial (ACC bits).
//   The column codes stay in the ADC output registers for a whole conversion window,
//   so UNITS datapaths are shared over the outputs: in slot s, datapath u updates
//   output s * UNITS + u (OUTS / UNITS slots per input bit).
//   The accumulators are not here: they live in UNITS SRAM banks of OUTS / UNITS words
//   (NVSim, characterize_shift_add.py). In slot s, datapath u reads word s of bank u
//   (acc_in) and the result (acc_out) is written back to it.
// Weights are unsigned 2-bit chunks (no signed-weight correction).
module shift_add #(
    parameter OUTS  = 32,
    parameter CELLS = 4,
    parameter CODE  = 7,
    parameter PART  = 14,
    parameter ACC   = 22,
    parameter UNITS = 4
) (
    input  wire                          first,
    input  wire [$clog2(OUTS/UNITS)-1:0] slot,
    input  wire [OUTS*CELLS*CODE-1:0]    codes,
    input  wire [UNITS*ACC-1:0]          acc_in,
    output wire [UNITS*ACC-1:0]          acc_out
);
    genvar u, k;
    generate
        for (u = 0; u < UNITS; u = u + 1) begin : datapath
            wire [PART-1:0] sum [0:CELLS];
            assign sum[0] = {PART{1'b0}};
            for (k = 0; k < CELLS; k = k + 1) begin : cell
                wire [CODE-1:0] c = codes[((slot * UNITS + u) * CELLS + k) * CODE +: CODE];
                assign sum[k + 1] = sum[k] + ({{(PART-CODE){1'b0}}, c} << (2 * k));
            end
            wire signed [ACC-1:0] p = $signed({{(ACC-PART){1'b0}}, sum[CELLS]});
            wire signed [ACC-1:0] a = acc_in[u * ACC +: ACC];
            assign acc_out[u * ACC +: ACC] = first ? -p : (a <<< 1) + p;
        end
    endgenerate
endmodule
