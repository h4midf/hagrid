// Per-token quantisation scale: from a row's abs-max M (unsigned W bits, the output of
// absmax_reduce_tree), the mantissa m and right shift sh of the factor 127 / M, so that
// custom_op_library/quantize/rtl/quant_mant_round.v gives q = clamp_int8(round(x * m *
// 2^-sh)) ~= round(x * 127 / M) for every x of the row. x and M are in the same fixed-
// point format, so the factor depends only on the integer M.
//
//   len = bit length of M (1..W)
//   sh  = min(len + MW - 8, SH_MAX)
//   m   = round(127 * 2^sh / M)       (integer divide, the rounding term M >> 1 added)
//
// 127 < 2^7 and 2^(len-1) <= M < 2^len put m in [127 * 2^(MW-8), 127 * 2^(MW-7)], i.e.
// [2032, 4064] at MW = 12: below 2^MW, with MW - 1 significant bits. When sh is capped at
// SH_MAX (len > SH_MAX - MW + 8) m is smaller, still below 2^MW. SH_MAX = 31 is
// quant_mant_round's own shift limit. M = 0 (an all-zero row) gives m = sh = 0 (q = 0).
// One evaluation per row.
module token_scale #(parameter W = 32, parameter MW = 12, parameter SH_MAX = 31) (
    input  wire [W-1:0]  absmax,
    output wire [MW-1:0] m,
    output wire [5:0]    sh
);
    // bit length of absmax: the position of its highest set bit, plus one
    reg [6:0] len;
    integer i;
    always @* begin
        len = 7'd0;
        for (i = 0; i < W; i = i + 1)
            if (absmax[i]) len = i + 1;
    end

    wire [6:0] sh_raw = len + (MW - 8);
    wire       zero   = (absmax == {W{1'b0}});
    wire [5:0] sh_c   = zero ? 6'd0 : ((sh_raw > SH_MAX) ? SH_MAX[5:0] : sh_raw[5:0]);

    // 127 * 2^SH_MAX + M / 2 fits in NW bits
    localparam NW = 7 + SH_MAX + 1;
    wire [NW-1:0] num  = ({{(NW - 7){1'b0}}, 7'd127} << sh_c) + (absmax >> 1);
    wire [NW-1:0] quot = zero ? {NW{1'b0}} : num / absmax;

    assign m  = quot[MW-1:0];
    assign sh = sh_c;
endmodule
