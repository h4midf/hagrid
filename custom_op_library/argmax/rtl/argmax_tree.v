// Comparator tree over N lanes of W-bit signed data: finds the maximum
// value and its index in one circuit, serving both cim.max (taps max_val)
// and cim.argmax (taps max_idx) per the two ops sharing one comparator-tree
// circuit. k=1 (plain max/argmax) only -- top-k is explicitly out of scope.
//
// N must be a power of two: all N lanes are real data supplied by the
// caller (the compiler's own tiling handles any count that isn't a full N
// upstream, same convention as reduce_tree.v). If a caller ever needs to
// pad a short row to N lanes, the pad value must be the most-negative
// W-bit value, never zero -- real data can be all-negative, so a zero pad
// would silently outrank it.
//
// Tie-break: strict '>' with the LEFT (lower-index) operand kept on ties,
// matching torch.argmax's own first-occurrence convention.
module argmax_tree #(parameter N = 8, parameter W = 32) (
    input  wire signed [N*W-1:0] x_flat,
    output wire signed [W-1:0]   max_val,
    output wire [$clog2(N)-1:0]  max_idx
);
    localparam LEVELS = $clog2(N);
    localparam IW = LEVELS;

    wire signed [W-1:0] val [0:LEVELS][0:N-1];
    wire        [IW-1:0] idx [0:LEVELS][0:N-1];

    genvar lvl, i;
    generate
        for (i = 0; i < N; i = i + 1) begin : leaf
            assign val[0][i] = x_flat[i*W +: W];
            assign idx[0][i] = i[IW-1:0];
        end
        for (lvl = 0; lvl < LEVELS; lvl = lvl + 1) begin : stage
            for (i = 0; i < (N >> (lvl + 1)); i = i + 1) begin : pair
                wire pick_left = val[lvl][2*i] >= val[lvl][2*i + 1];
                assign val[lvl + 1][i] = pick_left ? val[lvl][2*i] : val[lvl][2*i + 1];
                assign idx[lvl + 1][i] = pick_left ? idx[lvl][2*i] : idx[lvl][2*i + 1];
            end
        end
    endgenerate

    assign max_val = val[LEVELS][0];
    assign max_idx = idx[LEVELS][0];
endmodule
