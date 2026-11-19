// W-bit Kogge-Stone parallel-prefix adder: s = a + b + cin (W + 1 bits).
// The default yosys mapping of + is a ripple carry, too slow for one 1 GHz stage
// at 65 nm; the prefix network carries in log2(W) levels.
module sfu_prefix_add #(
    parameter W = 16
) (
    input  wire [W-1:0] a,
    input  wire [W-1:0] b,
    input  wire         cin,
    output wire [W:0]   s
);
    localparam L = $clog2(W);
    wire [W-1:0] g [0:L];
    wire [W-1:0] p [0:L];
    assign g[0] = a & b;
    assign p[0] = a ^ b;

    genvar k, i;
    generate
        for (k = 0; k < L; k = k + 1) begin : level
            for (i = 0; i < W; i = i + 1) begin : bit_
                if (i >= (1 << k)) begin : merge
                    assign g[k + 1][i] = g[k][i] | (p[k][i] & g[k][i - (1 << k)]);
                    assign p[k + 1][i] = p[k][i] & p[k][i - (1 << k)];
                end else begin : pass
                    assign g[k + 1][i] = g[k][i];
                    assign p[k + 1][i] = p[k][i];
                end
            end
        end
    endgenerate

    // carry into bit i + 1: the group (generate, propagate) of bits i..0 and cin
    wire [W:0] c;
    assign c[0] = cin;
    assign c[W:1] = g[L] | (p[L] & {W{cin}});
    assign s = {c[W], p[0] ^ c[W-1:0]};
endmodule
