// Wang et al. (IEEE TCAD 2025) Algorithm 3 nonuniform-LUT GELU: 24 bins (8 narrow "NC" +
// 16 wide "WD", split at a calibrated threshold T); x < -3 gives 0, x > 3 gives x.
// Hardwired constants (as softmax_row8.v's exp table), not a memory LUT. A step
// function: each bin outputs GELU at its center, so SLOPE is 0 for every segment.
//
// Breakpoints and outputs from bert-large-uncased layer-0 fc1 activations (T the
// 80/85/90/95th percentile with the least error), quantized with Eq. 3's power-of-two
// scale, separately at each width (two generate branches, Q12.20 and Q8.8).
module wang_tcad_gelu24 #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            localparam signed [31:0] DOM_LO = -32'sd3145728; // -3.0
            localparam signed [31:0] DOM_HI = 32'sd3145728;  // 3.0

            reg [4:0] idx;
            always @* begin
                if (x < -32'sd2916352) idx = 5'd0;
                else if (x < -32'sd2654208) idx = 5'd1;
                else if (x < -32'sd2424832) idx = 5'd2;
                else if (x < -32'sd2195456) idx = 5'd3;
                else if (x < -32'sd1933312) idx = 5'd4;
                else if (x < -32'sd1703936) idx = 5'd5;
                else if (x < -32'sd1474560) idx = 5'd6;
                else if (x < -32'sd1212416) idx = 5'd7;
                else if (x < -32'sd950272)  idx = 5'd8;
                else if (x < -32'sd688128)  idx = 5'd9;
                else if (x < -32'sd393216)  idx = 5'd10;
                else if (x < -32'sd131072)  idx = 5'd11;
                else if (x < 32'sd131072)   idx = 5'd12;
                else if (x < 32'sd425984)   idx = 5'd13;
                else if (x < 32'sd688128)   idx = 5'd14;
                else if (x < 32'sd950272)   idx = 5'd15;
                else if (x < 32'sd1245184)  idx = 5'd16;
                else if (x < 32'sd1507328)  idx = 5'd17;
                else if (x < 32'sd1769472)  idx = 5'd18;
                else if (x < 32'sd2064384)  idx = 5'd19;
                else if (x < 32'sd2326528)  idx = 5'd20;
                else if (x < 32'sd2588672)  idx = 5'd21;
                else if (x < 32'sd2883584)  idx = 5'd22;
                else                        idx = 5'd23;
            end

            reg signed [31:0] out;
            always @* begin
                case (idx)
                    5'd0: out = 32'sd0;         5'd1: out = 32'sd0;
                    5'd2: out = -32'sd32768;     5'd3: out = -32'sd32768;
                    5'd4: out = -32'sd65536;     5'd5: out = -32'sd65536;
                    5'd6: out = -32'sd98304;     5'd7: out = -32'sd131072;
                    5'd8: out = -32'sd163840;    5'd9: out = -32'sd163840;
                    5'd10: out = -32'sd163840;   5'd11: out = -32'sd98304;
                    5'd12: out = 32'sd0;         5'd13: out = 32'sd163840;
                    5'd14: out = 32'sd393216;    5'd15: out = 32'sd655360;
                    5'd16: out = 32'sd950272;    5'd17: out = 32'sd1245184;
                    5'd18: out = 32'sd1540096;   5'd19: out = 32'sd1867776;
                    5'd20: out = 32'sd2162688;   5'd21: out = 32'sd2424832;
                    5'd22: out = 32'sd2719744;   default: out = 32'sd3014656;
                endcase
            end

            assign y = (x < DOM_LO) ? 32'sd0 : (x > DOM_HI) ? x : out;
        end else if (W == 16 && FRAC == 8) begin : q8_8
            localparam signed [15:0] DOM_LO = -16'sd768;
            localparam signed [15:0] DOM_HI = 16'sd768;

            reg [4:0] idx;
            always @* begin
                if (x < -16'sd712) idx = 5'd0;
                else if (x < -16'sd648) idx = 5'd1;
                else if (x < -16'sd592) idx = 5'd2;
                else if (x < -16'sd536) idx = 5'd3;
                else if (x < -16'sd472) idx = 5'd4;
                else if (x < -16'sd416) idx = 5'd5;
                else if (x < -16'sd360) idx = 5'd6;
                else if (x < -16'sd296) idx = 5'd7;
                else if (x < -16'sd232) idx = 5'd8;
                else if (x < -16'sd168) idx = 5'd9;
                else if (x < -16'sd96)  idx = 5'd10;
                else if (x < -16'sd32)  idx = 5'd11;
                else if (x < 16'sd32)   idx = 5'd12;
                else if (x < 16'sd104)  idx = 5'd13;
                else if (x < 16'sd168)  idx = 5'd14;
                else if (x < 16'sd232)  idx = 5'd15;
                else if (x < 16'sd304)  idx = 5'd16;
                else if (x < 16'sd368)  idx = 5'd17;
                else if (x < 16'sd432)  idx = 5'd18;
                else if (x < 16'sd504)  idx = 5'd19;
                else if (x < 16'sd568)  idx = 5'd20;
                else if (x < 16'sd632)  idx = 5'd21;
                else if (x < 16'sd704)  idx = 5'd22;
                else                    idx = 5'd23;
            end

            reg signed [15:0] out;
            always @* begin
                case (idx)
                    5'd0: out = 16'sd0;    5'd1: out = 16'sd0;
                    5'd2: out = -16'sd8;    5'd3: out = -16'sd8;
                    5'd4: out = -16'sd16;   5'd5: out = -16'sd16;
                    5'd6: out = -16'sd24;   5'd7: out = -16'sd32;
                    5'd8: out = -16'sd40;   5'd9: out = -16'sd40;
                    5'd10: out = -16'sd40;  5'd11: out = -16'sd24;
                    5'd12: out = 16'sd0;    5'd13: out = 16'sd40;
                    5'd14: out = 16'sd96;   5'd15: out = 16'sd160;
                    5'd16: out = 16'sd232;  5'd17: out = 16'sd304;
                    5'd18: out = 16'sd376;  5'd19: out = 16'sd456;
                    5'd20: out = 16'sd528;  5'd21: out = 16'sd592;
                    5'd22: out = 16'sd664;  default: out = 16'sd736;
                endcase
            end

            assign y = (x < DOM_LO) ? 16'sd0 : (x > DOM_HI) ? x : out;
        end
    endgenerate
endmodule
