// Abs-max reduction tree over N signed W-bit lanes: absmax = max_i |x_i|, the row
// maximum of per-token (dynamic, one scale per row) int8 quantisation. Same binary
// tree as custom_op_library/softmax/rtl/max_reduce_tree.v (N a power of two), with
// |x| at each leaf and unsigned comparators. The output is unsigned W bits, so
// |-2^(W-1)| = 2^(W-1) is exact. A row wider than N is reduced in ceil(width / N)
// passes by the caller.
module absmax_reduce_tree #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    output wire        [W-1:0]   absmax
);
    localparam LEVELS = $clog2(N);

    wire [W-1:0] level [0:LEVELS][0:N-1];

    genvar lvl, i;
    generate
        for (i = 0; i < N; i = i + 1) begin : leaf
            wire [W-1:0] xi = x_flat[i*W +: W];
            // two's complement magnitude, read as unsigned W bits
            assign level[0][i] = xi[W-1] ? (~xi + 1'b1) : xi;
        end
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : stage
            for (i = 0; i < (N >> (lvl + 1)); i = i + 1) begin : pair
                assign level[lvl + 1][i] =
                    (level[lvl][2 * i] > level[lvl][2 * i + 1]) ? level[lvl][2 * i] : level[lvl][2 * i + 1];
            end
        end
    endgenerate

    assign absmax = level[LEVELS][0];
endmodule
