// Test harness for ailayernorm_stream (simulation only, never synthesized). One row is
// read from x.hex, alpha.hex, gamma.hex and beta.hex in the working directory (P = ROW / N
// words of N lanes each, lane 0 in the low bits) and streamed in every pass at the
// address the circuit's own `step` gives. rst is high in cycle 0, start in cycle 1.
module ailayernorm_stream_tb #(parameter N = 4, parameter W = 32, parameter FRAC = 20,
                               parameter ROW = 768, parameter EPS = 1) (
    input  wire                  clk,
    output wire signed [N*W-1:0] y_flat,
    output wire                  y_valid,
    output wire [1:0]            phase,
    output wire [$clog2(ROW / N)-1:0] step
);
    localparam P = ROW / N;
    reg [N*W-1:0] xm [0:P-1];
    reg [N*8-1:0] am [0:P-1];
    reg [N*W-1:0] gm [0:P-1];
    reg [N*W-1:0] bm [0:P-1];
    initial begin
        $readmemh("x.hex", xm);
        $readmemh("alpha.hex", am);
        $readmemh("gamma.hex", gm);
        $readmemh("beta.hex", bm);
    end

    reg [1:0] cnt = 2'd0;
    always @(posedge clk) if (cnt != 2'd3) cnt <= cnt + 2'd1;

    ailayernorm_stream #(.N(N), .W(W), .FRAC(FRAC), .ROW(ROW)) dut (
        .clk(clk), .rst(cnt == 2'd0), .start(cnt == 2'd1),
        .x_flat(xm[step]), .alpha_flat(am[step]), .gamma_flat(gm[step]), .beta_flat(bm[step]),
        .eps(EPS[W-1:0]), .phase(phase), .step(step), .y_flat(y_flat), .y_valid(y_valid));
endmodule
