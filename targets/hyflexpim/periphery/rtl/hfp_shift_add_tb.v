// Test harness for hfp_shift_add (simulation only, never synthesized): 8 input bits,
// each one array read whose 128 column codes arrive one per cycle (cycle c: bit c / 128,
// column c % 128). The code of column j at bit b is (j * 37 + b * 11 + SEED) mod 2^CODE
// (CODE = 6 for SLC, 7 for MLC). The output register is modeled here: 32 accumulators,
// read combinationally at idx and written when we is set.
module hfp_shift_add_tb #(
    parameter MODE = 1, parameter SEED = 5, parameter ACC = 22
) (
    input  wire              clk,
    output wire [32*ACC-1:0] acc_flat
);
    localparam CODE = MODE ? 7 : 6;
    reg [15:0] cnt = 16'd0;
    always @(posedge clk) cnt <= cnt + 16'd1;

    wire [15:0] bit_idx = cnt / 16'd128;
    wire [6:0]  col = cnt[6:0];
    wire [31:0] v = col * 37 + bit_idx * 11 + SEED;
    wire [6:0]  code = v[CODE-1:0];

    reg  [ACC-1:0] acc_mem [0:31];
    wire           we;
    wire [4:0]     idx;
    wire [ACC-1:0] acc_out;

    hfp_shift_add dut (
        .clk(clk), .valid(bit_idx < 16'd8), .mode(MODE[0]), .first(bit_idx == 16'd0),
        .col(col), .code(code), .acc_in(acc_mem[idx]), .we(we), .idx(idx), .acc_out(acc_out));

    always @(posedge clk)
        if (we) acc_mem[idx] <= acc_out;

    genvar o;
    generate
        for (o = 0; o < 32; o = o + 1) begin : flat
            assign acc_flat[o * ACC +: ACC] = acc_mem[o];
        end
    endgenerate
endmodule
