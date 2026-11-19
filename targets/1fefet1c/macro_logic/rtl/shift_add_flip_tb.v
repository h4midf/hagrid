// Test harness for shift_add_flip (simulation only, never synthesized). Cycle 0 loads
// the flip bits (column j flipped when (j * 7 + SEED) mod 3 == 0); then 8 input bits,
// each with OUTS / UNITS slots, one per cycle (cycle 1 + c: bit c / slots, slot
// c % slots). Codes an array can produce: the unit column's is U = (b * 13 + SEED)
// mod (ROWS + 1) (at most ROWS rows have input bit 1); column j's is
// v = (j * 37 + b * 11 + SEED) mod 2^CODE, and v mod (3 U + 1) if it is flipped (a
// flipped column sums at most 3 U). The accumulator SRAM banks are modeled as a
// register array (read in the slot, written at its end).
module shift_add_flip_tb #(
    parameter OUTS = 32, parameter CELLS = 4, parameter CODE = 8, parameter CV = 9,
    parameter PART = 16, parameter ACC = 24, parameter UNITS = 4, parameter ROWS = 128,
    parameter SEED = 5
) (
    input  wire                clk,
    output wire [OUTS*ACC-1:0] acc_flat
);
    localparam SLOTS = OUTS / UNITS;
    localparam SW = $clog2(SLOTS);
    localparam COLS = OUTS * CELLS;

    reg [15:0] cnt = 16'd0;
    always @(posedge clk) cnt <= cnt + 16'd1;

    wire        loading = (cnt == 16'd0);
    wire [15:0] step = cnt - 16'd1;
    wire [15:0] bit_idx = step / SLOTS;
    wire [SW-1:0] slot = step % SLOTS;

    wire [31:0] uv = (bit_idx * 13 + SEED) % (ROWS + 1);
    wire [COLS-1:0] flip_in;
    wire [COLS*CODE-1:0] codes;
    genvar j;
    generate
        for (j = 0; j < COLS; j = j + 1) begin : col
            wire [31:0] v = (j * 37 + bit_idx * 11 + SEED) % (1 << CODE);
            wire [31:0] vf = v % (3 * uv + 1);
            assign flip_in[j] = ((j * 7 + SEED) % 3) == 0;
            assign codes[j * CODE +: CODE] = flip_in[j] ? vf[CODE-1:0] : v[CODE-1:0];
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
                if (!loading && bit_idx < 16'd8)
                    acc[slot * UNITS + u] <= acc_out[u * ACC +: ACC];
            end
        end
        for (o = 0; o < OUTS; o = o + 1) begin : out
            assign acc_flat[o * ACC +: ACC] = acc[o];
        end
    endgenerate

    shift_add_flip #(.OUTS(OUTS), .CELLS(CELLS), .CODE(CODE), .CV(CV), .PART(PART), .ACC(ACC),
                     .UNITS(UNITS)) dut (
        .clk(clk), .load_flip(loading), .flip_in(flip_in),
        .first(bit_idx == 16'd0), .slot(slot), .codes(codes), .ucode(uv[CODE-1:0]),
        .acc_in(acc_in), .acc_out(acc_out)
    );
endmodule
