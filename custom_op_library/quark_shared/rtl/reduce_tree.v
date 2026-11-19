// Adder-tree reduction over N lanes of W-bit signed data: sum = x[0]+x[1]+...+x[N-1].
// N must be a power of two. No saturation: the output is widened by $clog2(N) bits,
// so a partial sum loses nothing before it is accumulated with other tiles.
module reduce_tree #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    output wire signed [W+$clog2(N)-1:0] sum
);
    localparam LEVELS = $clog2(N);
    localparam EXT = W + LEVELS;

    wire signed [EXT-1:0] level [0:LEVELS][0:N-1];

    genvar lvl, i;
    generate
        for (i = 0; i < N; i = i + 1) begin : leaf
            assign level[0][i] = {{LEVELS{x_flat[i*W+W-1]}}, x_flat[i*W +: W]};
        end
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : stage
            for (i = 0; i < (N >> (lvl + 1)); i = i + 1) begin : pair
                assign level[lvl + 1][i] = level[lvl][2 * i] + level[lvl][2 * i + 1];
            end
        end
    endgenerate

    assign sum = level[LEVELS][0];
endmodule
