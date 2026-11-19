// exp(x) for x <= 0, Q12.20 fixed point: softmax_row8.v's technique at Q12.20. Base-2
// decomposition: a shift for the integer power of 2, a 16-segment piecewise-linear
// function for the fraction, with its own Q12.20 table:
// OFFSET[k] = round(2^(-k/16) * 2^20), SLOPE[k] = round(2^(-(k+1)/16) * 2^20) - OFFSET[k].
// Error vs. true exp over x in [-20, 0]: mean 7.85e-6, max 2.29e-4
// (elu/approximation_error.py).
// CMOS logic, not memory: the 16 pairs are hardwired constants selected by a 16-way
// mux into one multiply-add.
module exp_pwl_q12_20 (
    input  wire signed [31:0] x,       // Q12.20, x <= 0
    output reg  signed [31:0] exp_out  // Q12.20 approximation of exp(x), in [0, 1]
);
    localparam signed [31:0] LOG2E_Q = 32'sd1512775; // round(log2(e) * 2^20)
    localparam FRAC = 20;
    localparam N_MAX = 21; // 2^-21 rounds to 0 at Q12.20; beyond this exp_out saturates to 0

    wire signed [63:0] y_full = x * LOG2E_Q;
    wire signed [63:0] y      = y_full >>> FRAC;
    wire signed [63:0] neg_y  = -y;             // >= 0, since x <= 0 and LOG2E_Q > 0

    wire [43:0] n_raw  = neg_y[63:20];          // integer part (number of halvings)
    wire [19:0] r_frac = neg_y[19:0];           // fractional part, FRAC bits
    wire [3:0]  seg    = r_frac[19:16];
    wire [15:0] intra  = r_frac[15:0];

    // 22 bits, not 21: OFFSET[0] = 2^20 = 1048576 needs a 22nd bit to stay
    // positive in signed two's complement (21 bits tops out at +1048575 and
    // 1048576 would wrap to -1048576).
    reg signed [21:0] off;
    reg signed [21:0] slope;
    always @* begin
        case (seg)
            4'd0:  begin off = 22'sd1048576; slope = -22'sd44456; end
            4'd1:  begin off = 22'sd1004120; slope = -22'sd42572; end
            4'd2:  begin off = 22'sd961548;  slope = -22'sd40766; end
            4'd3:  begin off = 22'sd920782;  slope = -22'sd39038; end
            4'd4:  begin off = 22'sd881744;  slope = -22'sd37383; end
            4'd5:  begin off = 22'sd844361;  slope = -22'sd35798; end
            4'd6:  begin off = 22'sd808563;  slope = -22'sd34281; end
            4'd7:  begin off = 22'sd774282;  slope = -22'sd32827; end
            4'd8:  begin off = 22'sd741455;  slope = -22'sd31435; end
            4'd9:  begin off = 22'sd710020;  slope = -22'sd30103; end
            4'd10: begin off = 22'sd679917;  slope = -22'sd28826; end
            4'd11: begin off = 22'sd651091;  slope = -22'sd27604; end
            4'd12: begin off = 22'sd623487;  slope = -22'sd26434; end
            4'd13: begin off = 22'sd597053;  slope = -22'sd25313; end
            4'd14: begin off = 22'sd571740;  slope = -22'sd24240; end
            4'd15: begin off = 22'sd547500;  slope = -22'sd23212; end
        endcase
    end

    wire signed [38:0] prod   = slope * $signed({1'b0, intra});
    wire signed [31:0] g_full = off + (prod >>> 16);

    always @* begin
        if (n_raw >= N_MAX)
            exp_out = 32'sd0;
        else
            exp_out = g_full >>> n_raw[4:0];
    end
endmodule
