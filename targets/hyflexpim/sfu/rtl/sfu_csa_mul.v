// Unsigned A x B multiplier up to the carry-save form: the B partial products
// (a AND b[i], shifted by i), plus `addend` when C = 1, reduced by a Wallace tree of
// 3:2 compressors to two rows, sum + carry = a * b (+ addend) modulo 2^(A + B). The
// final sum is a separate adder (sfu_prefix_add), so a pipeline register can sit
// between them.
// Each level compresses every group of three rows into two; the rows left over pass
// through. R rows take about log1.5(R / 2) levels (11 rows: 5 levels).
module sfu_csa_mul #(
    parameter A = 11,
    parameter B = 11,
    parameter C = 0
) (
    input  wire [A-1:0]   a,
    input  wire [B-1:0]   b,
    input  wire [A+B-1:0] addend,
    output wire [A+B-1:0] sum,
    output wire [A+B-1:0] carry
);
    localparam W = A + B;
    localparam R = B + C;

    // rows at level k: n_{k+1} = 2 floor(n_k / 3) + n_k mod 3
    function integer rows_at(input integer k);
        integer n, j;
        begin
            n = R;
            for (j = 0; j < k; j = j + 1)
                n = 2 * (n / 3) + (n % 3);
            rows_at = n;
        end
    endfunction

    function integer levels(input integer dummy);
        integer n, j;
        begin
            n = R;
            j = 0;
            while (n > 2) begin
                n = 2 * (n / 3) + (n % 3);
                j = j + 1;
            end
            levels = j;
        end
    endfunction

    localparam L = levels(0);

    wire [W-1:0] row [0:L][0:R-1];

    genvar k, g, r;
    generate
        for (r = 0; r < B; r = r + 1) begin : pp
            assign row[0][r] = b[r] ? ({{B{1'b0}}, a} << r) : {W{1'b0}};
        end
        if (C) begin : add_row
            assign row[0][B] = addend;
        end
        for (k = 0; k < L; k = k + 1) begin : level
            localparam integer N = rows_at(k);
            localparam integer G = N / 3;
            for (g = 0; g < G; g = g + 1) begin : csa
                wire [W-1:0] x = row[k][3 * g];
                wire [W-1:0] y = row[k][3 * g + 1];
                wire [W-1:0] z = row[k][3 * g + 2];
                assign row[k + 1][2 * g]     = x ^ y ^ z;
                assign row[k + 1][2 * g + 1] = ((x & y) | (x & z) | (y & z)) << 1;
            end
            for (r = 0; r < N % 3; r = r + 1) begin : pass
                assign row[k + 1][2 * G + r] = row[k][3 * G + r];
            end
        end
    endgenerate

    assign sum   = row[L][0];
    assign carry = row[L][1];
endmodule
