// GQA-LUT (DAC 2024): an 8-entry GA-searched piecewise-linear GELU (Algorithm 1 with
// Rounding Mutation, Algorithm 2) over [-4, 4], a slope and intercept per segment.
// x is not clamped: an out-of-domain x falls in the first / last segment, whose line
// extrapolates (GELU ~ x for large x). Clamping instead raised bert-large's perplexity
// from 7.24 to 8.20 (baseline 7.36): ~45% of its fc1 pre-activations are outside [-4, 4].
//
// Breakpoints and intercepts are int8 with Eq. 3's shared power-of-two scale (the
// smallest keeping both in [-128, 127]); slopes stay full precision, as in the paper.
module gqa_lut_gelu8 #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            reg [2:0] idx;
            always @* begin
                if (x < -32'sd2883584) idx = 3'd0;
                else if (x < -32'sd2293760) idx = 3'd1;
                else if (x < -32'sd1146880) idx = 3'd2;
                else if (x < -32'sd786432)  idx = 3'd3;
                else if (x < -32'sd262144)  idx = 3'd4;
                else if (x < 32'sd491520)   idx = 3'd5;
                else if (x < 32'sd2359296)  idx = 3'd6;
                else                         idx = 3'd7;
            end

            reg signed [31:0] slope;
            reg signed [31:0] icept;
            always @* begin
                case (idx)
                    3'd0: begin slope = -32'sd5807;    icept = -32'sd32768;  end
                    3'd1: begin slope = -32'sd42184;   icept = -32'sd131072; end
                    3'd2: begin slope = -32'sd117730;  icept = -32'sd294912; end
                    3'd3: begin slope = -32'sd65924;   icept = -32'sd229376; end
                    3'd4: begin slope = 32'sd143396;   icept = -32'sd98304;  end
                    3'd5: begin slope = 32'sd610830;   icept = 32'sd0;       end
                    3'd6: begin slope = 32'sd1144232;  icept = -32'sd262144; end
                    default: begin slope = 32'sd1062282; icept = -32'sd65536; end
                endcase
            end

            localparam signed [63:0] HALF = 64'sd1 <<< (FRAC - 1);
            localparam signed [63:0] MAXV = (64'sd1 <<< 31) - 1;
            localparam signed [63:0] MINV = -(64'sd1 <<< 31);
            wire signed [63:0] p_full = $signed(slope) * $signed(x);
            wire signed [63:0] p_r    = (p_full + HALF) >>> FRAC;
            wire signed [63:0] y_full = p_r + $signed(icept);
            assign y = (y_full > MAXV) ? MAXV[31:0] : (y_full < MINV) ? MINV[31:0] : y_full[31:0];
        end else if (W == 16 && FRAC == 8) begin : q8_8
            reg [2:0] idx;
            always @* begin
                if (x < -16'sd704) idx = 3'd0;
                else if (x < -16'sd560) idx = 3'd1;
                else if (x < -16'sd280) idx = 3'd2;
                else if (x < -16'sd192) idx = 3'd3;
                else if (x < -16'sd64)  idx = 3'd4;
                else if (x < 16'sd120)  idx = 3'd5;
                else if (x < 16'sd576)  idx = 3'd6;
                else                     idx = 3'd7;
            end

            reg signed [15:0] slope;
            reg signed [15:0] icept;
            always @* begin
                case (idx)
                    3'd0: begin slope = -16'sd1;   icept = -16'sd8;  end
                    3'd1: begin slope = -16'sd10;  icept = -16'sd32; end
                    3'd2: begin slope = -16'sd29;  icept = -16'sd72; end
                    3'd3: begin slope = -16'sd16;  icept = -16'sd56; end
                    3'd4: begin slope = 16'sd35;   icept = -16'sd24; end
                    3'd5: begin slope = 16'sd149;  icept = 16'sd0;   end
                    3'd6: begin slope = 16'sd279;  icept = -16'sd64; end
                    default: begin slope = 16'sd259; icept = -16'sd16; end
                endcase
            end

            localparam signed [31:0] HALF = 32'sd1 <<< (FRAC - 1);
            localparam signed [31:0] MAXV = (32'sd1 <<< 15) - 1;
            localparam signed [31:0] MINV = -(32'sd1 <<< 15);
            wire signed [31:0] p_full = $signed(slope) * $signed(x);
            wire signed [31:0] p_r    = (p_full + HALF) >>> FRAC;
            wire signed [31:0] y_full = p_r + $signed(icept);
            assign y = (y_full > MAXV) ? MAXV[15:0] : (y_full < MINV) ? MINV[15:0] : y_full[15:0];
        end
    endgenerate
endmodule
