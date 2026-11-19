// Test harness for shift_add (simulation only, never synthesized): 8 input bits, each
// with OUTS / UNITS slots, one per cycle (cycle c: bit c / slots, slot c % slots). The
// code of column j at bit b is (j * 37 + b * 11 + SEED) mod 2^CODE. The accumulator
// SRAM banks are modeled as a register array (read in the slot, written at its end).
module shift_add_tb #(
    parameter OUTS = 32, parameter CELLS = 4, parameter CODE = 7,
    parameter PART = 14, parameter ACC = 22, parameter UNITS = 4, parameter SEED = 5
) (
    input  wire                clk,
    output wire [OUTS*ACC-1:0] acc_flat
);
    localparam SLOTS = OUTS / UNITS;
    localparam SW = $clog2(SLOTS);
    localparam COLS = OUTS * CELLS;

    reg [15:0] cnt = 16'd0;
    always @(posedge clk) cnt <= cnt + 16'd1;

    wire [15:0] bit_idx = cnt / SLOTS;
    wire [SW-1:0] slot = cnt % SLOTS;

    wire [COLS*CODE-1:0] codes;
    genvar j;
    generate
        for (j = 0; j < COLS; j = j + 1) begin : col
            wire [31:0] v = j * 37 + bit_idx * 11 + SEED;
            assign codes[j * CODE +: CODE] = v[CODE-1:0];
        end
    endgenerate

    // the accumulator SRAM banks: bank u holds outputs s * UNITS + u, word s
    reg  [ACC-1:0] acc [0:OUTS-1];
    wire [UNITS*ACC-1:0] acc_in, acc_out;
    genvar u, o;
    generate
        for (u = 0; u < UNITS; u = u + 1) begin : bank
            assign acc_in[u * ACC +: ACC] = acc[slot * UNITS + u];
            always @(posedge clk) begin
                if (bit_idx < 16'd8)
                    acc[slot * UNITS + u] <= acc_out[u * ACC +: ACC];
            end
        end
        for (o = 0; o < OUTS; o = o + 1) begin : out
            assign acc_flat[o * ACC +: ACC] = acc[o];
        end
    endgenerate

    shift_add #(.OUTS(OUTS), .CELLS(CELLS), .CODE(CODE), .PART(PART), .ACC(ACC), .UNITS(UNITS)) dut (
        .first(bit_idx == 16'd0), .slot(slot), .codes(codes), .acc_in(acc_in), .acc_out(acc_out)
    );
endmodule
