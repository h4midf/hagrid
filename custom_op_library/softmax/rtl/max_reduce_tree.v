// Max over N signed W-bit lanes: reduce_tree.v's binary tree (N a power of two)
// with a 2-input max comparator in place of each adder. The output stays W bits.
module max_reduce_tree #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    output wire signed [W-1:0] max_val
);
    localparam LEVELS = $clog2(N);

    wire signed [W-1:0] level [0:LEVELS][0:N-1];

    genvar lvl, i;
    generate
        for (i = 0; i < N; i = i + 1) begin : leaf
            assign level[0][i] = x_flat[i*W +: W];
        end
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : stage
            for (i = 0; i < (N >> (lvl + 1)); i = i + 1) begin : pair
                assign level[lvl + 1][i] =
                    (level[lvl][2 * i] > level[lvl][2 * i + 1]) ? level[lvl][2 * i] : level[lvl][2 * i + 1];
            end
        end
    endgenerate

    assign max_val = level[LEVELS][0];
endmodule
