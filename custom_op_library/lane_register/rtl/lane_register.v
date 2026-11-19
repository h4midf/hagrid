// Register of one shared 1-lane unit (vec_add, vec_el_mul, vec_select, per-token
// rounding, a per-element softmax stage): holds the lane's input operands while it computes and its result until the
// result is written back to the WMU. BITS = input bits + result bits. The operands
// come from the column ADCs' output registers after a WMU read (no other buffer).
module lane_register #(parameter BITS = 96) (
    input  wire            clk,
    input  wire            load,
    input  wire [BITS-1:0] d,
    output reg  [BITS-1:0] q
);
    always @(posedge clk) begin
        if (load)
            q <= d;
    end
endmodule
