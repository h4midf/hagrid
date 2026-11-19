// ISAAC's shift-and-add as its Table I sizes it: one W-bit shift-and-add, y = 2 acc + x
// (ISAAC computes in 16-bit fixed point), combinational, no registers. Synthesized only
// to check our flow against ISAAC's S+A row (0.00006 mm^2, 0.05 mW at 32 nm,
// 1.2 GHz), the number HyFlexPIM's Table 2 S&A row is scaled from (through ASADI).
module isaac_shift_add #(
    parameter W = 16
) (
    input  wire [W-1:0] acc,
    input  wire [W-1:0] x,
    output wire [W:0]   y
);
    sfu_prefix_add #(.W(W)) add (.a({acc[W-2:0], 1'b0}), .b(x), .cin(1'b0), .s(y));
endmodule
