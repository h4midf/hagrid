// Layer normalisation over one row of ROW elements, streamed N lanes per cycle. The
// squares are kept at 2W bits: clamping them to W bits would drop an outlier element's
// contribution to the variance (|d| > 45 at Q12.20, |d| > 11.3 at Q8.8). Per-element
// affine: gamma and beta are streamed with x in the last pass (one per element, as a
// LayerNorm's weight and bias).
//
// Schedule after start (P = ROW / N cycles per pass; x is streamed in every pass, the
// lane group of cycle `step` being elements step*N .. step*N+N-1):
//   SUM   P cycles  acc += sum(x lanes); at the end mean = (acc * RECIP) >>> RS
//   VAR   P cycles  d = clamp(x - mean), sq = round(d*d >> FRAC) at 2W bits,
//                   acc += sum(sq); at the end var = clamp((acc * RECIP) >>> RS),
//                   var_eps = clamp(var + eps)
//   RSQ   1 cycle   inv_std = rsqrt(var_eps)
//   NORM  P cycles  norm = round(d * inv_std >> FRAC), y = clamp(round(norm * gamma
//                   >> FRAC) + beta); y_valid high
// 3P + 1 cycles per row. One 2W-bit reduce_tree serves SUM and VAR. The N squaring
// multipliers are reused for d * inv_std in NORM, so the datapath has 2N multipliers.
//
// The division by ROW is a multiply by RECIP = round(2^RS / ROW) and an arithmetic
// shift (floor), so ROW need not be a power of two. mean is the low W bits; var is
// clamped to W bits.
// W=32/FRAC=20 (rsqrt_q12_20) or W=16/FRAC=8 (rsqrt_q8_8). ROW must be a multiple of
// N, N a power of two.
module layer_norm_stream #(parameter N = 4, parameter W = 32, parameter FRAC = 20,
                           parameter ROW = 768, parameter RS = 40) (
    input  wire                  clk,
    input  wire                  rst,
    input  wire                  start,
    input  wire signed [N*W-1:0] x_flat,
    input  wire signed [N*W-1:0] gamma_flat,
    input  wire signed [N*W-1:0] beta_flat,
    input  wire signed [W-1:0]   eps,
    output reg  [2:0]            phase,
    output reg  [$clog2(ROW / N)-1:0] step,
    output wire signed [N*W-1:0] y_flat,
    output wire                  y_valid
);
    localparam P = ROW / N;   // at least 2
    localparam LEVELS = $clog2(N);
    localparam AW = 2 * W + $clog2(ROW) + 1;             // row accumulator (sum of 2W-bit squares)
    localparam IDLE = 3'd0, SUM = 3'd1, VAR = 3'd2, RSQ = 3'd3, NORM = 3'd4;
    localparam signed [W:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0] MINV = -(1 <<< (W - 1));

    reg signed [AW-1:0] acc;
    reg signed [W-1:0]  mean;
    reg signed [W-1:0]  var_eps;
    reg signed [W-1:0]  inv_std;

    // ---- lanes (ln_stream_lane.v): the 2W-bit square in VAR, y in NORM
    wire signed [2*W*N-1:0] sq_flat;
    wire signed [2*W*N-1:0] xw_flat;   // x sign-extended to 2W bits for SUM
    genvar i;
    generate
        for (i = 0; i < N; i = i + 1) begin : lane
            ln_stream_lane #(.W(W), .FRAC(FRAC)) u_lane (
                .x(x_flat[i*W +: W]), .mean(mean), .inv_std(inv_std), .norm(phase == NORM),
                .gamma(gamma_flat[i*W +: W]), .beta(beta_flat[i*W +: W]),
                .sq(sq_flat[i*2*W +: 2*W]), .y(y_flat[i*W +: W]));
            wire signed [W-1:0] x_i = x_flat[i*W +: W];
            wire signed [2*W-1:0] x_w = x_i;
            assign xw_flat[i*2*W +: 2*W] = x_w;
        end
    endgenerate
    assign y_valid = (phase == NORM);

    // ---- one 2W-bit reduce_tree: x in SUM, the squares in VAR
    wire signed [2*W+LEVELS-1:0] lanes_sum;
    reduce_tree #(.N(N), .W(2 * W)) u_sum (.x_flat((phase == SUM) ? xw_flat : sq_flat), .sum(lanes_sum));
    wire signed [AW-1:0] acc_next = acc + $signed(lanes_sum);

    // ---- / ROW (row_div.v): mean = the low W bits, var = clamped to W bits
    wire signed [AW-1:0] row_q;
    row_div #(.AW(AW), .QW(AW), .ROW(ROW), .RS(RS)) u_div (.sum(acc_next), .q(row_q));
    wire signed [W-1:0] row_mean = row_q[W-1:0];
    localparam signed [AW-1:0] MAXA = MAXV;
    wire signed [W-1:0] row_var = (row_q > MAXA) ? MAXV[W-1:0] : row_q[W-1:0];   // var >= 0
    wire signed [W:0] ve_full = $signed(row_var) + $signed(eps);
    wire signed [W-1:0] ve = (ve_full > MAXV) ? MAXV[W-1:0] : (ve_full < MINV) ? MINV[W-1:0] : ve_full[W-1:0];

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
            acc <= 0;
        end else begin
            case (phase)
                IDLE: if (start) begin
                    phase <= SUM;
                    step <= 0;
                    acc <= 0;
                end
                SUM: begin
                    if (last) begin
                        mean <= row_mean;
                        acc <= 0;
                        step <= 0;
                        phase <= VAR;
                    end else begin
                        acc <= acc_next;
                        step <= step + 1'b1;
                    end
                end
                VAR: begin
                    if (last) begin
                        var_eps <= ve;
                        acc <= 0;
                        step <= 0;
                        phase <= RSQ;
                    end else begin
                        acc <= acc_next;
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
                default: phase <= IDLE;
            endcase
        end
    end
endmodule
