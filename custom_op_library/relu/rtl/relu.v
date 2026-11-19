// ReLU: y = max(x, 0). Exact. W=32 for Q12.20, W=16 for Q8.8.
module relu #(parameter W = 32) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    assign y = x[W-1] ? {W{1'b0}} : x;
endmodule
