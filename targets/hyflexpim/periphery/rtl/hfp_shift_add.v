// Shift-and-add (S&A) of one HyFlexPIM analog array (Table 2: one per array).
// HyFlexPIM Sec. 3.2, Fig. 6-8: the 128 bitline values are converted one after another
// by the array's shared SAR ADC (1.28 GS/s), so the S&A takes one code per conversion:
//   SLC (mode 0): 1-bit cells, 6-bit codes; a weight spans 8 adjacent columns, the
//                 column at position k weighs 2^k (Fig. 6).
//   MLC (mode 1): 2-bit cells, 7-bit codes; a weight spans 4 columns, weight 4^k (Fig. 7).
// A weight's partial sum is accumulated over its columns in carry-save form (no carry
// chain in the 1.28 GHz loop), then added; inputs arrive one bit per array read, MSB
// first, two's complement: acc = -partial on the sign bit, then acc = 2 acc + partial.
// The accumulators are the array's output register (Table 2 'OR', a separate row): the
// S&A reads acc_in and writes acc_out at output idx when we is set.
// Weights are unsigned bit chunks (Fig. 6 / 7), as in the paper's S&A equations.
// Stages: A carry-save accumulation over a weight's columns (loop), B partial = sum +
// carry, C accumulator update.
module hfp_shift_add #(
    parameter PART = 14,          // partial sum bits (SLC 63 * 255, MLC 127 * 85 < 2^14)
    parameter ACC  = 22           // accumulator bits (|acc| <= 128 * max partial < 2^21)
) (
    input  wire            clk,
    input  wire            valid,         // a code this cycle
    input  wire            mode,          // 0 SLC, 1 MLC
    input  wire            first,         // the input's sign bit (first array read)
    input  wire [6:0]      col,           // column of this code, 0..127
    input  wire [6:0]      code,
    input  wire [ACC-1:0]  acc_in,        // OR[idx]
    output wire            we,
    output wire [4:0]      idx,
    output wire [ACC-1:0]  acc_out
);
    // ---- stage A: term = code * 2^k (SLC) or 4^k (MLC); carry-save sum over the weight
    wire [2:0]      k     = mode ? {1'b0, col[1:0]} : col[2:0];
    wire [2:0]      shift = mode ? {col[1:0], 1'b0} : col[2:0];
    wire            start = (k == 3'd0);
    wire            last  = mode ? (col[1:0] == 2'd3) : (col[2:0] == 3'd7);
    wire [4:0]      o     = mode ? col[6:2] : {1'b0, col[6:3]};
    wire [PART-1:0] term  = {{(PART-7){1'b0}}, code} << shift;

    reg  [PART-1:0] ps, pc;
    wire [PART-1:0] xs = start ? {PART{1'b0}} : ps;
    wire [PART-1:0] xc = start ? {PART{1'b0}} : pc;
    wire [PART-1:0] s_n = xs ^ xc ^ term;
    wire [PART-1:0] c_n = ((xs & xc) | (xs & term) | (xc & term)) << 1;
    always @(posedge clk)
        if (valid) begin
            ps <= s_n;
            pc <= c_n;
        end

    // the completed weight's sum and carry, on its last column
    reg  [PART-1:0] bs, bc;
    reg  [4:0]      bo;
    reg             bfirst, bv;
    always @(posedge clk) begin
        bv <= valid & last;
        if (valid & last) begin
            bs <= s_n;
            bc <= c_n;
            bo <= o;
            bfirst <= first;
        end
    end

    // ---- stage B: partial = sum + carry
    wire [PART:0] psum;
    sfu_prefix_add #(.W(PART)) add_b (.a(bs), .b(bc), .cin(1'b0), .s(psum));
    reg  [PART-1:0] cp;
    reg  [4:0]      co;
    reg             cfirst, cv;
    always @(posedge clk) begin
        cv <= bv;
        cp <= psum[PART-1:0];
        co <= bo;
        cfirst <= bfirst;
    end

    // ---- stage C: acc = -partial (sign bit) or 2 acc + partial
    wire [ACC-1:0] p_ext = {{(ACC-PART){1'b0}}, cp};
    wire [ACC:0]   upd;
    sfu_prefix_add #(.W(ACC)) add_c (
        .a(cfirst ? ~p_ext : {acc_in[ACC-2:0], 1'b0}),
        .b(cfirst ? {ACC{1'b0}} : p_ext),
        .cin(cfirst), .s(upd));
    assign acc_out = upd[ACC-1:0];
    assign we      = cv;
    assign idx     = co;
endmodule
