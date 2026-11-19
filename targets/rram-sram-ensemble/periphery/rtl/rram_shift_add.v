// Shift-and-add of one read-out circuit of the RRAM macro (Krishnan et al., TCAD 2022,
// Sec. III-A): each 3-bit ADC code is accumulated by its input bit's significance. Inputs
// arrive MSB first and are unsigned: acc = code on the first bit, then acc = 2 acc + code.
// The accumulators are the macro's output buffer (acc_in / acc_out).
// ACC = 9: up to 6-bit activations (Sec. V), 7 * 63 < 2^9.
module rram_shift_add #(
    parameter ACC = 9
) (
    input  wire           first,
    input  wire [2:0]     code,
    input  wire [ACC-1:0] acc_in,
    output wire [ACC-1:0] acc_out
);
    wire [ACC-1:0] base = first ? {ACC{1'b0}} : {acc_in[ACC-2:0], 1'b0};
    assign acc_out = base + {{(ACC-3){1'b0}}, code};
endmodule
