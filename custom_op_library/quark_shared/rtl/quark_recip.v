// Newton-Raphson reciprocal: y ~= 1/x for x > 0, y_{i+1} = y_i*(2-x*y_i), 2 iterations.
// quark_shared_div needs it for exp(y) = 1/exp(-y) when y > 0 (quark_appro_exp only
// takes y <= 0), so it is hardware beyond QUARK's shared exp/ln circuit. Seeded by
// leading-one detection (as rsqrt_q12_20.v) and a 32-entry table of 1/m, m in [1,2).
//
// x must not be so small that y0 = seed << (-e) overflows W bits (roughly
// msb_pos(x) >= 2*FRAC-W+1); its inputs (shared_div's Newton iterate, LayerNorm's
// std_inv) stay far above that.
module quark_recip #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,   // x > 0 required
    output wire signed [W-1:0] y    // approx 1/x
);
    localparam IDX_BITS = 5;
    localparam signed [W-1:0] ONE = 1 <<< FRAC;
    localparam signed [W:0]   MAXV = (1 <<< (W - 1)) - 1;
    localparam signed [W:0]   MINV = -(1 <<< (W - 1));
    localparam signed [2*W-1:0] MAXV2 = MAXV;
    localparam signed [2*W-1:0] MINV2 = MINV;
    localparam signed [2*W-1:0] HALF = {{(2*W-1){1'b0}}, 1'b1} <<< (FRAC - 1);
    localparam signed [W-1:0]   TWO  = 2 <<< FRAC;

    function [$clog2(W)-1:0] msb_pos;
        input [W-1:0] v;
        integer k;
        reg found;
        begin
            msb_pos = 0;
            found = 1'b0;
            for (k = W - 1; k >= 0; k = k - 1) begin
                if (!found && v[k]) begin
                    msb_pos = k[$clog2(W)-1:0];
                    found = 1'b1;
                end
            end
        end
    endfunction

    wire [W-1:0] xu = x[W-1:0]; // x > 0 by contract
    wire [$clog2(W)-1:0] p = msb_pos(xu);
    wire signed [W-1:0]  p_s = {{(W-$clog2(W)){1'b0}}, p};
    wire signed [W-1:0]  e   = p_s - FRAC; // real exponent: x_real ~= 2^e * m, m in [1,2)

    wire [IDX_BITS-1:0] idx = (p >= IDX_BITS) ? ((xu >> (p - IDX_BITS)) & {IDX_BITS{1'b1}})
                                               : ((xu << (IDX_BITS - p)) & {IDX_BITS{1'b1}});

    // seed[idx] = round((1/(1+idx/32)) * 2^FRAC)
    reg signed [W-1:0] seed;
    always @* begin
        case (idx)
            5'd0:  seed = (W==32)?32'sd1048576:16'sd256;  5'd1:  seed = (W==32)?32'sd1016801:16'sd248;
            5'd2:  seed = (W==32)?32'sd986895:16'sd241;    5'd3:  seed = (W==32)?32'sd958698:16'sd234;
            5'd4:  seed = (W==32)?32'sd932068:16'sd228;    5'd5:  seed = (W==32)?32'sd906877:16'sd221;
            5'd6:  seed = (W==32)?32'sd883011:16'sd216;    5'd7:  seed = (W==32)?32'sd860370:16'sd210;
            5'd8:  seed = (W==32)?32'sd838861:16'sd205;    5'd9:  seed = (W==32)?32'sd818401:16'sd200;
            5'd10: seed = (W==32)?32'sd798915:16'sd195;    5'd11: seed = (W==32)?32'sd780336:16'sd191;
            5'd12: seed = (W==32)?32'sd762601:16'sd186;    5'd13: seed = (W==32)?32'sd745654:16'sd182;
            5'd14: seed = (W==32)?32'sd729444:16'sd178;    5'd15: seed = (W==32)?32'sd713924:16'sd174;
            5'd16: seed = (W==32)?32'sd699051:16'sd171;    5'd17: seed = (W==32)?32'sd684784:16'sd167;
            5'd18: seed = (W==32)?32'sd671089:16'sd164;    5'd19: seed = (W==32)?32'sd657930:16'sd161;
            5'd20: seed = (W==32)?32'sd645278:16'sd158;    5'd21: seed = (W==32)?32'sd633102:16'sd155;
            5'd22: seed = (W==32)?32'sd621378:16'sd152;    5'd23: seed = (W==32)?32'sd610081:16'sd149;
            5'd24: seed = (W==32)?32'sd599186:16'sd146;    5'd25: seed = (W==32)?32'sd588674:16'sd144;
            5'd26: seed = (W==32)?32'sd578525:16'sd141;    5'd27: seed = (W==32)?32'sd568719:16'sd139;
            5'd28: seed = (W==32)?32'sd559241:16'sd137;    5'd29: seed = (W==32)?32'sd550073:16'sd134;
            5'd30: seed = (W==32)?32'sd541201:16'sd132;    5'd31: seed = (W==32)?32'sd532610:16'sd130;
        endcase
    end

    // y0 = seed >> e (scale the [0.5,1]-range table entry by 2^-e)
    wire signed [W-1:0] y0 = (e >= 0) ? (seed >>> e) : (seed <<< (-e));

    // Newton iteration: y_{i+1} = y_i*(2 - x*y_i), 2 iterations unrolled.
    wire signed [2*W-1:0] xy0_full = $signed(x) * $signed(y0);
    wire signed [2*W-1:0] xy0_r    = (xy0_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] t0_full  = TWO - xy0_r;
    wire signed [W-1:0]   t0 = (t0_full > MAXV2) ? MAXV[W-1:0] : (t0_full < MINV2) ? MINV[W-1:0] : t0_full[W-1:0];
    wire signed [2*W-1:0] y1_full  = $signed(y0) * $signed(t0);
    wire signed [2*W-1:0] y1_r     = (y1_full + HALF) >>> FRAC;
    wire signed [W-1:0]   y1 = (y1_r > MAXV2) ? MAXV[W-1:0] : (y1_r < MINV2) ? MINV[W-1:0] : y1_r[W-1:0];

    wire signed [2*W-1:0] xy1_full = $signed(x) * $signed(y1);
    wire signed [2*W-1:0] xy1_r    = (xy1_full + HALF) >>> FRAC;
    wire signed [2*W-1:0] t1_full  = TWO - xy1_r;
    wire signed [W-1:0]   t1 = (t1_full > MAXV2) ? MAXV[W-1:0] : (t1_full < MINV2) ? MINV[W-1:0] : t1_full[W-1:0];
    wire signed [2*W-1:0] y2_full  = $signed(y1) * $signed(t1);
    wire signed [2*W-1:0] y2_r     = (y2_full + HALF) >>> FRAC;
    assign y = (y2_r > MAXV2) ? MAXV[W-1:0] : (y2_r < MINV2) ? MINV[W-1:0] : y2_r[W-1:0];
endmodule
