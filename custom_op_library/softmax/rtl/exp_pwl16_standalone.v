// One element of softmax_row8.v's exp_pwl16 (RMSL's base-2 decomposition + 16-segment
// PWL exp, bit for bit), on its own so a row of any width is costed per element
// after max_reduce_tree.
module exp_pwl16_standalone (
    input  signed [23:0] x,            // Q8.8 (in a wide container), x <= 0, natural-log delta from the row max
    output reg  signed [15:0] exp_out  // Q8.8 approximation of exp(x), in [0, 1]
);
    localparam signed [15:0] LOG2E_Q8_8 = 16'sd369;
    localparam N_MAX = 9;

    wire signed [39:0] y_full = x * LOG2E_Q8_8;
    wire signed [31:0] y      = y_full[39:8];
    wire signed [31:0] neg_y  = -y;
    wire [23:0] n_raw   = neg_y[31:8];
    wire [7:0]  r_frac  = neg_y[7:0];
    wire [3:0]  seg      = r_frac[7:4];
    wire [3:0]  intra    = r_frac[3:0];

    reg signed [9:0] off;
    reg signed [4:0] slope;
    always @* begin
        case (seg)
            4'd0:  begin off = 10'sd256; slope = -5'sd11; end
            4'd1:  begin off = 9'sd245; slope = -5'sd10; end
            4'd2:  begin off = 9'sd235; slope = -5'sd10; end
            4'd3:  begin off = 9'sd225; slope = -5'sd10; end
            4'd4:  begin off = 9'sd215; slope = -5'sd9;  end
            4'd5:  begin off = 9'sd206; slope = -5'sd9;  end
            4'd6:  begin off = 9'sd197; slope = -5'sd8;  end
            4'd7:  begin off = 9'sd189; slope = -5'sd8;  end
            4'd8:  begin off = 9'sd181; slope = -5'sd8;  end
            4'd9:  begin off = 9'sd173; slope = -5'sd7;  end
            4'd10: begin off = 9'sd166; slope = -5'sd7;  end
            4'd11: begin off = 9'sd159; slope = -5'sd7;  end
            4'd12: begin off = 9'sd152; slope = -5'sd6;  end
            4'd13: begin off = 9'sd146; slope = -5'sd6;  end
            4'd14: begin off = 9'sd140; slope = -5'sd6;  end
            4'd15: begin off = 9'sd134; slope = -5'sd6;  end
        endcase
    end

    wire signed [15:0] g_full = (off <<< 4) + slope * $signed({1'b0, intra});
    wire signed [11:0] g_r    = g_full[15:4];

    always @* begin
        if (n_raw >= N_MAX)
            exp_out = 16'sd0;
        else
            exp_out = $signed(g_r) >>> n_raw[3:0];
    end
endmodule
