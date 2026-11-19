// One pipeline stage register of the SFU units. EN = 0 makes it a wire, so the
// tests can evaluate a unit's logic combinationally (yosys eval) with the same
// netlist the synthesis registers.
module sfu_pipe #(
    parameter W  = 1,
    parameter EN = 1
) (
    input  wire         clk,
    input  wire [W-1:0] d,
    output wire [W-1:0] q
);
    generate
        if (EN) begin : reg_stage
            reg [W-1:0] r;
            always @(posedge clk) r <= d;
            assign q = r;
        end else begin : wire_stage
            assign q = d;
        end
    endgenerate
endmodule
