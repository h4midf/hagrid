// Shifted-ReLU (HAMMER, IEEE CAL 2023), its exp replacement in softmax:
// y = relu(x) + relu(x-1) + relu(x-2) + relu(x-3), with 1 = ONE = 1 <<< FRAC.
// The paper gives no closed form or bit widths, so this is our reconstruction.
// Each diff is 2 bits wider to avoid wraparound near MINV, and the sum is 2
// more bits wider before the final saturate. W=32/FRAC=20 for Q12.20,
// W=16/FRAC=8 for Q8.8.
module shifted_relu #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    localparam signed [W-1:0] ONE = 1 <<< FRAC;
    localparam signed [W+1:0] MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W+1:0] MINV = -(1 <<< (W - 1));

    wire signed [W+1:0] xx = {{2{x[W-1]}}, x};
    wire signed [W+1:0] d0 = xx;
    wire signed [W+1:0] d1 = xx - ONE;
    wire signed [W+1:0] d2 = xx - 2 * ONE;
    wire signed [W+1:0] d3 = xx - 3 * ONE;

    wire signed [W+1:0] r0 = d0[W+1] ? {(W+2){1'b0}} : d0;
    wire signed [W+1:0] r1 = d1[W+1] ? {(W+2){1'b0}} : d1;
    wire signed [W+1:0] r2 = d2[W+1] ? {(W+2){1'b0}} : d2;
    wire signed [W+1:0] r3 = d3[W+1] ? {(W+2){1'b0}} : d3;

    wire signed [W+3:0] sum = {{2{r0[W+1]}}, r0} + {{2{r1[W+1]}}, r1} + {{2{r2[W+1]}}, r2} + {{2{r3[W+1]}}, r3};

    localparam signed [W+3:0] OUT_MAXV = MAXV;
    localparam signed [W+3:0] OUT_MINV = MINV;
    assign y = (sum > OUT_MAXV) ? MAXV[W-1:0] : (sum < OUT_MINV) ? MINV[W-1:0] : sum[W-1:0];
endmodule
