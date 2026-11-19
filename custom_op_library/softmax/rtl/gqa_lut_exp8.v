// GQA-LUT (DAC'24) 8-entry piecewise-linear exp over [-8, 0] (x <= 0 after
// max subtraction). Like gqa_lut_gelu8.v, but x below -8 outputs 0 instead of
// extrapolating the first segment: 4-9% of bert-large attention scores fall below
// -8, and extrapolating there gave 5.7e4 ppl (gqa_lut_bert_large.py's gqa_pwl_exp()).
// Breakpoints and intercepts are INT8-quantized with Eq.(3)'s shared power-of-two
// scale; slopes keep full precision.
module gqa_lut_exp8 #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,   // x <= 0 required
    output wire signed [W-1:0] y
);
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            localparam signed [31:0] DOM_LO = -32'sd8388608; // -8.0

            reg [2:0] idx;
            always @* begin
                if (x < -32'sd6160384) idx = 3'd0;
                else if (x < -32'sd3866624) idx = 3'd1;
                else if (x < -32'sd2686976) idx = 3'd2;
                else if (x < -32'sd1769472) idx = 3'd3;
                else if (x < -32'sd1048576) idx = 3'd4;
                else if (x < -32'sd589824)  idx = 3'd5;
                else if (x < -32'sd262144)  idx = 3'd6;
                else                         idx = 3'd7;
            end

            reg signed [31:0] slope;
            reg signed [31:0] icept;
            always @* begin
                case (idx)
                    3'd0: begin slope = 32'sd1128;   icept = 32'sd0;       end
                    3'd1: begin slope = 32'sd9839;   icept = 32'sd65536;   end
                    3'd2: begin slope = 32'sd47971;  icept = 32'sd196608;  end
                    3'd3: begin slope = 32'sd127934; icept = 32'sd393216;  end
                    3'd4: begin slope = 32'sd275262; icept = 32'sd655360;  end
                    3'd5: begin slope = 32'sd480592; icept = 32'sd851968;  end
                    3'd6: begin slope = 32'sd699413; icept = 32'sd983040;  end
                    default: begin slope = 32'sd922184; icept = 32'sd1048576; end
                endcase
            end

            localparam signed [63:0] HALF = 64'sd1 <<< (FRAC - 1);
            localparam signed [63:0] MAXV = (64'sd1 <<< 31) - 1;
            localparam signed [63:0] MINV = -(64'sd1 <<< 31);
            wire signed [63:0] p_full = $signed(slope) * $signed(x);
            wire signed [63:0] p_r    = (p_full + HALF) >>> FRAC;
            wire signed [63:0] y_full = p_r + $signed(icept);
            wire signed [31:0] y_lut  = (y_full > MAXV) ? MAXV[31:0] : (y_full < MINV) ? MINV[31:0] : y_full[31:0];
            assign y = (x < DOM_LO) ? 32'sd0 : y_lut;
        end else if (W == 16 && FRAC == 8) begin : q8_8
            localparam signed [15:0] DOM_LO = -16'sd2048; // -8.0

            reg [2:0] idx;
            always @* begin
                if (x < -16'sd1504) idx = 3'd0;
                else if (x < -16'sd944) idx = 3'd1;
                else if (x < -16'sd656) idx = 3'd2;
                else if (x < -16'sd432) idx = 3'd3;
                else if (x < -16'sd256) idx = 3'd4;
                else if (x < -16'sd144) idx = 3'd5;
                else if (x < -16'sd64)  idx = 3'd6;
                else                     idx = 3'd7;
            end

            reg signed [15:0] slope;
            reg signed [15:0] icept;
            always @* begin
                case (idx)
                    3'd0: begin slope = 16'sd0;   icept = 16'sd0;   end
                    3'd1: begin slope = 16'sd2;   icept = 16'sd16;  end
                    3'd2: begin slope = 16'sd12;  icept = 16'sd48;  end
                    3'd3: begin slope = 16'sd31;  icept = 16'sd96;  end
                    3'd4: begin slope = 16'sd67;  icept = 16'sd160; end
                    3'd5: begin slope = 16'sd117; icept = 16'sd208; end
                    3'd6: begin slope = 16'sd171; icept = 16'sd240; end
                    default: begin slope = 16'sd225; icept = 16'sd256; end
                endcase
            end

            localparam signed [31:0] HALF = 32'sd1 <<< (FRAC - 1);
            localparam signed [31:0] MAXV = (32'sd1 <<< 15) - 1;
            localparam signed [31:0] MINV = -(32'sd1 <<< 15);
            wire signed [31:0] p_full = $signed(slope) * $signed(x);
            wire signed [31:0] p_r    = (p_full + HALF) >>> FRAC;
            wire signed [31:0] y_full = p_r + $signed(icept);
            wire signed [15:0] y_lut  = (y_full > MAXV) ? MAXV[15:0] : (y_full < MINV) ? MINV[15:0] : y_full[15:0];
            assign y = (x < DOM_LO) ? 16'sd0 : y_lut;
        end
    endgenerate
endmodule
