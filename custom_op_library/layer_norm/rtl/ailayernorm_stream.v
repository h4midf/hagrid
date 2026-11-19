// SOLE's AILayerNorm (ICCAD 2023) over one row of ROW elements, streamed N lanes per
// cycle: per-lane PTF quantize, DynamicCompress, Square&Decompress, the <<4 correction,
// a 1-LSB variance floor and rsqrt. The decompressed squares are rescaled by
// 2*alpha + FRAC (Q.FRAC units, like E(x)^2), and the squares, E(x^2) and E(x)^2 are
// kept at 2W bits with var clamped to W bits only after the subtraction. alpha (8-bit
// signed PTF exponent), gamma and beta are per element, streamed with x.
//
// var = E[x_shifted-compressed^2] - E[x_shifted]^2 needs no mean first, so both sums
// accumulate in one pass. Schedule after start (P = ROW / N cycles per pass; x and
// alpha are streamed in both passes, the lane group of cycle `step` being elements
// step*N .. step*N+N-1):
//   STATS P cycles  sx += sum(x_shifted lanes), sq += sum(xc2_shifted lanes); at the
//                   end ex = sx / ROW (low W bits), ex2 = (sq / ROW) << 4 (wide)
//   RSQ   1 cycle   inv_std = rsqrt(var_eps) (ailn_var.v)
//   NORM  P cycles  y (ailn_stream_lane.v); y_valid high
// 2P + 1 cycles per row. / ROW is row_div.v (a multiply by round(2^RS / ROW) and an
// arithmetic shift).
// Submodules: ailn_stream_lane (N), reduce_tree (2), row_div (2), ailn_var, rsqrt.
// W=32/FRAC=20 (rsqrt_q12_20) or W=16/FRAC=8 (rsqrt_q8_8). ROW must be a multiple of
// N, N a power of two.
module ailayernorm_stream #(parameter N = 4, parameter W = 32, parameter FRAC = 20,
                            parameter ROW = 768, parameter RS = 40) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [N*8-1:0] alpha_flat,
    input  wire signed [N*W-1:0] gamma_flat,
    input  wire signed [N*W-1:0] beta_flat,
    input  wire signed [W-1:0]   eps,
    output reg  [1:0]            phase,
    output reg  [$clog2(ROW / N)-1:0] step,
    output wire signed [N*W-1:0] y_flat,
    output wire                  y_valid
);
    localparam P = ROW / N;   // at least 2
    localparam LEVELS = $clog2(N);
    localparam AW = W + $clog2(ROW) + 1;        // sum of x_shifted
    localparam AW2 = 2 * W + $clog2(ROW) + 1;   // sum of the 2W-bit squares
    localparam EW = AW2 + 4;                    // E(x^2) after the <<4
    localparam IDLE = 2'd0, STATS = 2'd1, RSQ = 2'd2, NORM = 2'd3;

    reg signed [AW-1:0]  sx;
    reg signed [AW2-1:0] sq;
    reg signed [EW-1:0]  ex2;
    reg signed [W-1:0]   ex, inv_std;

    wire [W*N-1:0] xs_flat;
    wire [2*W*N-1:0] xc2s_flat;
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            ailn_stream_lane #(.W(W), .FRAC(FRAC)) u_lane (
                .x(x_flat[i*W +: W]), .alpha(alpha_flat[i*8 +: 8]), .ex(ex), .inv_std(inv_std),
                .gamma(gamma_flat[i*W +: W]), .beta(beta_flat[i*W +: W]),
                .x_shifted(xs_flat[i*W +: W]), .xc2_shifted(xc2s_flat[i*2*W +: 2*W]), .y(y_flat[i*W +: W]));
        end
    endgenerate
    assign y_valid = (phase == NORM);

    // ---- STATS: two reduce_trees, two row accumulators, / ROW
    wire signed [W+LEVELS-1:0] lanes_sx;
    wire signed [2*W+LEVELS-1:0] lanes_sq;
    reduce_tree #(.N(N), .W(W)) u_sx (.x_flat(xs_flat), .sum(lanes_sx));
    reduce_tree #(.N(N), .W(2 * W)) u_sq (.x_flat(xc2s_flat), .sum(lanes_sq));
    wire signed [AW-1:0] sx_next = sx + $signed(lanes_sx);
    wire signed [AW2-1:0] sq_next = sq + $signed(lanes_sq);
    wire signed [W-1:0] ex_new;
    row_div #(.AW(AW), .QW(W), .ROW(ROW), .RS(RS)) u_div_x (.sum(sx_next), .q(ex_new));
    wire signed [AW2-1:0] qq;
    row_div #(.AW(AW2), .QW(AW2), .ROW(ROW), .RS(RS)) u_div_q (.sum(sq_next), .q(qq));
    wire signed [EW-1:0] ex2_new = qq <<< 4;   // the averaging-step <<4 correction, wide

    // ---- RSQ
    wire signed [W-1:0] var_eps;
    ailn_var #(.W(W), .FRAC(FRAC), .EW(EW)) u_var (.ex(ex), .ex2(ex2), .eps(eps), .var_eps(var_eps));
    wire signed [W-1:0] rsqrt_val;
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            rsqrt_q12_20 u_rsqrt (.val(var_eps), .rsqrt_out(rsqrt_val));
        end else if (W == 16 && FRAC == 8) begin : q8_8
            rsqrt_q8_8 u_rsqrt (.val(var_eps), .rsqrt_out(rsqrt_val));
        end
    endgenerate

    wire last = (step == P - 1);
    always @(posedge clk) begin
        if (rst) begin
            phase <= IDLE;
            step <= 0;
            sx <= 0;
            sq <= 0;
        end else begin
            case (phase)
                IDLE: if (start) begin
                    phase <= STATS;
                    step <= 0;
                    sx <= 0;
                    sq <= 0;
                end
                STATS: begin
                    if (last) begin
                        ex <= ex_new;
                        ex2 <= ex2_new;
                        step <= 0;
                        phase <= RSQ;
                    end else begin
                        sx <= sx_next;
                        sq <= sq_next;
                        step <= step + 1'b1;
                    end
                end
                RSQ: begin
                    inv_std <= rsqrt_val;
                    phase <= NORM;
                end
                NORM: begin
                    if (last) begin
                        step <= 0;
                        phase <= IDLE;
                    end else begin
                        step <= step + 1'b1;
                    end
                end
            endcase
        end
    end
endmodule
