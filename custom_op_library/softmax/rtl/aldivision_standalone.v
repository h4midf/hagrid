// e2softmax_row8.v's aldivision_q8_8 (SOLE's Eq.9-13: S = 2^k_s*(1+s) and a
// bias-corrected 1-bit linear approximation of 1/(1+s), no divider), with the sum
// width SUM_W as a parameter (a row of N terms sums to up to N*256).
module aldivision_standalone #(parameter SUM_W = 20) (
    input  [3:0] k_y,
    input  signed [SUM_W-1:0] s_sum,   // Q8.8 row sum, > 0 (sum of >= 1 nonzero exp terms)
    output reg signed [15:0] p_out
);
    localparam PBIT_W = $clog2(SUM_W);

    function [PBIT_W-1:0] msb_pos;
        input [SUM_W-1:0] v;
        integer i;
        reg found;
        begin
            msb_pos = {PBIT_W{1'b0}};
            found = 1'b0;
            for (i = SUM_W - 1; i >= 0; i = i - 1) begin
                if (!found && v[i]) begin
                    msb_pos = i[PBIT_W-1:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [SUM_W-1:0] s_u   = s_sum[SUM_W-1:0]; // s_sum > 0 by contract, safe to treat as unsigned
    wire [PBIT_W-1:0] p_bit = msb_pos(s_u);
    // Widened by plain assignment, not concatenation (which is unsigned), before
    // the signed subtract of FRAC = 8.
    wire signed [PBIT_W:0] p_bit_w = p_bit;
    localparam signed [PBIT_W:0] FRAC_8 = 8;
    wire signed [PBIT_W:0] k_s = p_bit_w - FRAC_8;
    wire next_bit = (p_bit == {PBIT_W{1'b0}}) ? 1'b0 : s_u[p_bit - 1'b1];

    localparam signed [15:0] CONST_HI = 16'sd419; // round(1.636*256), q=0
    localparam signed [15:0] CONST_LO = 16'sd291; // round(1.136*256), q=0.5
    wire signed [15:0] const_sel = next_bit ? CONST_LO : CONST_HI;

    wire signed [8:0] k_y_w = k_y;
    wire signed [8:0] k_s_w = k_s;
    wire signed [8:0] ks_sum = k_y_w + k_s_w + 9'sd1;

    always @* begin
        if (ks_sum >= 9'sd0) begin
            if (ks_sum >= 9'sd16)
                p_out = 16'sd0;
            else
                p_out = const_sel >>> ks_sum[3:0];
        end else begin
            if (ks_sum <= -9'sd16)
                p_out = 16'sd32767;
            else
                p_out = const_sel <<< (-ks_sum[3:0]);
        end
    end
endmodule
