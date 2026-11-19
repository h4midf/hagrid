// 1/sqrt(val) for val > 0, Q12.20 fixed point. Leading-one detection gives the
// exponent p_real = msb(val) - 20, split into e = floor(p_real/2) and its parity
// bit. Two 32-entry seed tables (one per parity, mantissa m in [1,2) at 5-bit
// resolution) avoid folding a sqrt(2) factor into a shift. The seed is shifted by e
// (right for e >= 0, left for e < 0) and refined by one Newton-Raphson iteration,
// y1 = y0*(1.5 - 0.5*val*y0*y0).
//
// Max relative error 0.15% over val in [1e-6, 4000] (layer_norm/approximation_error.py);
// a second iteration does not improve it. CMOS logic, not memory: 64 hardwired seed
// constants behind a mux.
module rsqrt_q12_20 (
    input  wire signed [31:0] val,       // Q12.20, val > 0
    output wire signed [31:0] rsqrt_out  // Q12.20 approximation of 1/sqrt(val)
);
    localparam FRAC = 20;
    localparam IDX_BITS = 5;
    localparam signed [31:0] ONE_HALF_Q = 32'sd1572864; // round(1.5 * 2^20)

    function [4:0] msb_pos;
        input [31:0] v;
        integer i;
        reg found;
        begin
            msb_pos = 5'd0;
            found = 1'b0;
            for (i = 31; i >= 0; i = i - 1) begin
                if (!found && v[i]) begin
                    msb_pos = i[4:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [31:0] uval = val; // caller guarantees val > 0
    wire [4:0]  p = msb_pos(uval);
    wire signed [6:0] p_real = {2'b00, p} - FRAC;
    wire signed [6:0] e = p_real >>> 1;    // floor(p_real / 2)
    wire parity = p_real[0];               // floor-mod-2 residue, via two's complement

    wire [4:0] idx = (p >= IDX_BITS) ? ((uval >> (p - IDX_BITS)) & 5'h1F)
                                      : ((uval << (IDX_BITS - p)) & 5'h1F);

    reg signed [31:0] seed;
    always @* begin
        case ({parity, idx})
            6'd0: seed = 32'sd1048576;
            6'd1: seed = 32'sd1032566;
            6'd2: seed = 32'sd1017268;
            6'd3: seed = 32'sd1002630;
            6'd4: seed = 32'sd988607;
            6'd5: seed = 32'sd975156;
            6'd6: seed = 32'sd962239;
            6'd7: seed = 32'sd949823;
            6'd8: seed = 32'sd937875;
            6'd9: seed = 32'sd926367;
            6'd10: seed = 32'sd915272;
            6'd11: seed = 32'sd904567;
            6'd12: seed = 32'sd894229;
            6'd13: seed = 32'sd884237;
            6'd14: seed = 32'sd874573;
            6'd15: seed = 32'sd865219;
            6'd16: seed = 32'sd856159;
            6'd17: seed = 32'sd847377;
            6'd18: seed = 32'sd838861;
            6'd19: seed = 32'sd830596;
            6'd20: seed = 32'sd822571;
            6'd21: seed = 32'sd814774;
            6'd22: seed = 32'sd807194;
            6'd23: seed = 32'sd799822;
            6'd24: seed = 32'sd792649;
            6'd25: seed = 32'sd785665;
            6'd26: seed = 32'sd778863;
            6'd27: seed = 32'sd772234;
            6'd28: seed = 32'sd765772;
            6'd29: seed = 32'sd759469;
            6'd30: seed = 32'sd753319;
            6'd31: seed = 32'sd747317;
            6'd32: seed = 32'sd741455;
            6'd33: seed = 32'sd730135;
            6'd34: seed = 32'sd719317;
            6'd35: seed = 32'sd708967;
            6'd36: seed = 32'sd699051;
            6'd37: seed = 32'sd689539;
            6'd38: seed = 32'sd680406;
            6'd39: seed = 32'sd671626;
            6'd40: seed = 32'sd663178;
            6'd41: seed = 32'sd655040;
            6'd42: seed = 32'sd647195;
            6'd43: seed = 32'sd639625;
            6'd44: seed = 32'sd632315;
            6'd45: seed = 32'sd625250;
            6'd46: seed = 32'sd618416;
            6'd47: seed = 32'sd611802;
            6'd48: seed = 32'sd605396;
            6'd49: seed = 32'sd599186;
            6'd50: seed = 32'sd593164;
            6'd51: seed = 32'sd587320;
            6'd52: seed = 32'sd581645;
            6'd53: seed = 32'sd576132;
            6'd54: seed = 32'sd570772;
            6'd55: seed = 32'sd565560;
            6'd56: seed = 32'sd560487;
            6'd57: seed = 32'sd555549;
            6'd58: seed = 32'sd550739;
            6'd59: seed = 32'sd546052;
            6'd60: seed = 32'sd541482;
            6'd61: seed = 32'sd537026;
            6'd62: seed = 32'sd532677;
            6'd63: seed = 32'sd528433;
        endcase
    end

    // For a tiny val (1 LSB gives rsqrt ~1024) y0 fits 32 bits but y0^2 does not, so
    // every intermediate is kept wide and only narrowed to 32 bits at the end.
    wire signed [31:0] y0 = (e >= 0) ? (seed >>> e) : (seed <<< (-e));

    wire signed [63:0]  yy_full  = y0 * y0;
    wire signed [63:0]  yy_q     = yy_full >>> FRAC;
    wire signed [127:0] vyy_full = $signed(val) * yy_q;
    wire signed [127:0] vyy_q    = vyy_full >>> FRAC;
    wire signed [127:0] term_q   = ONE_HALF_Q - (vyy_q >>> 1);
    wire signed [127:0] y1_full  = $signed(y0) * term_q;
    wire signed [127:0] y1       = y1_full >>> FRAC;

    localparam signed [127:0] OUT_MAXV = (128'sd1 <<< 31) - 1;
    localparam signed [127:0] OUT_MINV = -(128'sd1 <<< 31);
    assign rsqrt_out = (y1 > OUT_MAXV) ? OUT_MAXV[31:0] : (y1 < OUT_MINV) ? OUT_MINV[31:0] : y1[31:0];
endmodule
