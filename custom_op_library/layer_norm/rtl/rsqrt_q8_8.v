// 1/sqrt(val) for val > 0, Q8.8 fixed point: rsqrt_q12_20.v's scheme at Q8.8
// (p_real = msb(val) - 8, two 32-entry seed tables, one Newton-Raphson iteration).
//
// Max relative error 7.8% over the LayerNorm variance range [0.01, 20]
// (approximation_error.py); a second iteration only reaches 6.3%, limited by the
// 5-bit seed table and Q8.8 rounding. CMOS logic, not memory: 64 hardwired seed
// constants behind a mux.
module rsqrt_q8_8 (
    input  wire signed [15:0] val,       // Q8.8, val > 0
    output wire signed [15:0] rsqrt_out  // Q8.8 approximation of 1/sqrt(val)
);
    localparam FRAC = 8;
    localparam IDX_BITS = 5;
    localparam signed [15:0] ONE_HALF_Q = 16'sd384; // round(1.5 * 2^8)

    function [3:0] msb_pos;
        input [15:0] v;
        integer i;
        reg found;
        begin
            msb_pos = 4'd0;
            found = 1'b0;
            for (i = 15; i >= 0; i = i - 1) begin
                if (!found && v[i]) begin
                    msb_pos = i[3:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [15:0] uval = val;
    wire [3:0]  p = msb_pos(uval);
    wire signed [5:0] p_real = {2'b00, p} - FRAC;
    wire signed [5:0] e = p_real >>> 1;
    wire parity = p_real[0];

    wire [4:0] idx = (p >= IDX_BITS) ? ((uval >> (p - IDX_BITS)) & 5'h1F)
                                      : ((uval << (IDX_BITS - p)) & 5'h1F);

    reg signed [15:0] seed;
    always @* begin
        case ({parity, idx})
            6'd0: seed = 16'sd256;
            6'd1: seed = 16'sd252;
            6'd2: seed = 16'sd248;
            6'd3: seed = 16'sd245;
            6'd4: seed = 16'sd241;
            6'd5: seed = 16'sd238;
            6'd6: seed = 16'sd235;
            6'd7: seed = 16'sd232;
            6'd8: seed = 16'sd229;
            6'd9: seed = 16'sd226;
            6'd10: seed = 16'sd223;
            6'd11: seed = 16'sd221;
            6'd12: seed = 16'sd218;
            6'd13: seed = 16'sd216;
            6'd14: seed = 16'sd214;
            6'd15: seed = 16'sd211;
            6'd16: seed = 16'sd209;
            6'd17: seed = 16'sd207;
            6'd18: seed = 16'sd205;
            6'd19: seed = 16'sd203;
            6'd20: seed = 16'sd201;
            6'd21: seed = 16'sd199;
            6'd22: seed = 16'sd197;
            6'd23: seed = 16'sd195;
            6'd24: seed = 16'sd194;
            6'd25: seed = 16'sd192;
            6'd26: seed = 16'sd190;
            6'd27: seed = 16'sd189;
            6'd28: seed = 16'sd187;
            6'd29: seed = 16'sd185;
            6'd30: seed = 16'sd184;
            6'd31: seed = 16'sd182;
            6'd32: seed = 16'sd181;
            6'd33: seed = 16'sd178;
            6'd34: seed = 16'sd176;
            6'd35: seed = 16'sd173;
            6'd36: seed = 16'sd171;
            6'd37: seed = 16'sd168;
            6'd38: seed = 16'sd166;
            6'd39: seed = 16'sd164;
            6'd40: seed = 16'sd162;
            6'd41: seed = 16'sd160;
            6'd42: seed = 16'sd158;
            6'd43: seed = 16'sd156;
            6'd44: seed = 16'sd154;
            6'd45: seed = 16'sd153;
            6'd46: seed = 16'sd151;
            6'd47: seed = 16'sd149;
            6'd48: seed = 16'sd148;
            6'd49: seed = 16'sd146;
            6'd50: seed = 16'sd145;
            6'd51: seed = 16'sd143;
            6'd52: seed = 16'sd142;
            6'd53: seed = 16'sd141;
            6'd54: seed = 16'sd139;
            6'd55: seed = 16'sd138;
            6'd56: seed = 16'sd137;
            6'd57: seed = 16'sd136;
            6'd58: seed = 16'sd134;
            6'd59: seed = 16'sd133;
            6'd60: seed = 16'sd132;
            6'd61: seed = 16'sd131;
            6'd62: seed = 16'sd130;
            6'd63: seed = 16'sd129;
        endcase
    end

    wire signed [15:0] y0 = (e >= 0) ? (seed >>> e) : (seed <<< (-e));

    wire signed [31:0] yy_full  = y0 * y0;
    wire signed [31:0] yy_q     = yy_full >>> FRAC;
    wire signed [63:0] vyy_full = $signed(val) * yy_q;
    wire signed [63:0] vyy_q    = vyy_full >>> FRAC;
    wire signed [63:0] term_q   = ONE_HALF_Q - (vyy_q >>> 1);
    wire signed [63:0] y1_full  = $signed(y0) * term_q;
    wire signed [63:0] y1       = y1_full >>> FRAC;

    localparam signed [63:0] OUT_MAXV = (64'sd1 <<< 15) - 1;
    localparam signed [63:0] OUT_MINV = -(64'sd1 <<< 15);
    assign rsqrt_out = (y1 > OUT_MAXV) ? OUT_MAXV[15:0] : (y1 < OUT_MINV) ? OUT_MINV[15:0] : y1[15:0];
endmodule
