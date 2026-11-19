// One lane of the HyFlexPIM SFU. HyFlexPIM Sec. 3.1: the SFU handles softmax, layer
// normalization and GELU with 'max search, subtraction, exponentiation (via Taylor
// series), addition, division, multiplication, and square root functions, all fully
// pipelined using floating-point arithmetic. After computation, the results are
// converted to the integer format'; 'each SFU [processes] 256 inputs per cycle':
// 256 lanes. FP16 (Sec. 5.1).
// One fully pipelined unit for max, subtraction, addition, division, multiplication and
// square root, and the FP16 -> int8 conversion. Exponentiation has no unit of its own:
// as in the authors' prior work (TransPIM: five-order Taylor series computed with
// multiplications and additions; PIM-GPT: Taylor terms on shared FP adders and
// multipliers), it runs on the lane's multiplier, adder, subtractor and converter
// (fp16_ref.exp), with two small units for its range reduction: int8 -> FP16 (n back
// to FP16) and x 2^n (ldexp). Each unit's result is registered; the multiplier, adder,
// subtractor and FP16 -> int8 also register their operands (the operand multiplexers
// and those units' first stages do not fit one 1 GHz cycle together).
// Exp mode (exp_mode = 1): after rst, one exp starts every 7 cycles (x on exp_x at
// phase 0, the first cycle after rst) and leaves on ldexp_y 71 cycles later; the multiplier is busy every cycle (7 per exp).
// Static modulo schedule, the one needing the least storage (unit latencies, operand
// register included: mul 4, add / sub 6, to_int8 2, int8_to_fp16 1, ldexp 1); each
// operation's start cycle s, phase s mod 7:
//   mul  t = x log2e   s 0  (ph 0)    to_int8  n = round(t)  s 4  (ph 4)
//   i2f  n -> FP16     s 6  (ph 6)    sub      f = t - n     s 7  (ph 0)
//   mul  r = f ln2     s 13 (ph 6)    mul      p = c5 r      s 17 (ph 3)
//   add  p + c4        s 21 (ph 0)    mul      p r           s 30 (ph 2)
//   add  p + c3        s 34 (ph 6)    mul      p r           s 40 (ph 5)
//   add  p + c2        s 44 (ph 2)    mul      p r           s 50 (ph 1)
//   add  p + c1        s 54 (ph 5)    mul      p r           s 60 (ph 4)
//   add  p + c0        s 64 (ph 1)    ldexp    p 2^n         s 70 (ph 0)
// Values kept: t (s 4 to 7) and p (s 27 to 30) in one register each, r (s 17 to 60) in
// 7 rotating slots and n (s 6 to 70) in 10, slot = exp index mod 7 or mod 10.
// Not here (not described in the paper): the control that issues operations outside
// exp, accumulation registers, and cross-lane reduction.
module sfu_lane (
    input  wire        clk,
    input  wire        rst,               // synchronous; starts the exp schedule at phase 0
    input  wire        exp_mode,
    input  wire [15:0] exp_x,
    input  wire [15:0] max_a, max_b,
    input  wire [15:0] sub_a, sub_b,
    input  wire [15:0] add_a, add_b,
    input  wire [15:0] div_a, div_b,
    input  wire [15:0] mul_a, mul_b,
    input  wire [15:0] sqrt_a,
    input  wire [15:0] int_a,
    input  wire [7:0]  i2f_n,
    input  wire [15:0] ldexp_a,
    input  wire [7:0]  ldexp_n,
    output reg  [15:0] max_y, sub_y, add_y, div_y, mul_y, sqrt_y, i2f_y, ldexp_y,
    output reg  [7:0]  int_y
);
    // FP16 constants (fp16_ref.EXP_LOG2E, EXP_LN2, EXP_TAYLOR)
    localparam [15:0] LOG2E = 16'h3DC5, LN2 = 16'h398C;
    localparam [15:0] C0 = 16'h3C00, C1 = 16'h3C00, C2 = 16'h3800,
                      C3 = 16'h3155, C4 = 16'h2955, C5 = 16'h2044;

    // ---- exp schedule state, one-hot so every select comes straight from a register:
    // phase ring (bit p set in phase p), exp index E mod 7 and mod 10
    reg  [6:0] ph;
    reg  [6:0] e7;
    reg  [9:0] e10;
    always @(posedge clk) begin
        if (rst) begin
            ph  <= 7'b0000001;
            e7  <= 7'b0000001;
            e10 <= 10'b0000000001;
        end else begin
            ph <= {ph[5:0], ph[6]};
            if (ph[6]) begin
                e7  <= {e7[5:0], e7[6]};
                e10 <= {e10[8:0], e10[9]};
            end
        end
    end

    // stored values; slot s holds exp E - d when E mod K = (s + d) mod K
    reg  [15:0] t_hold, p_hold;
    reg  [15:0] r_slot [0:6];
    reg  [7:0]  n_slot [0:9];
    always @(posedge clk) begin
        if (exp_mode && ph[4]) t_hold <= mul_y;                          // t, s 4
        if (exp_mode && ph[6]) p_hold <= add_y;                          // p, s 27
    end
    genvar g;
    generate
        for (g = 0; g < 7; g = g + 1) begin : r_wr
            always @(posedge clk)
                if (exp_mode && ph[3] && e7[(g + 2) % 7]) r_slot[g] <= mul_y;   // r of exp E - 2
        end
        for (g = 0; g < 10; g = g + 1) begin : n_wr
            always @(posedge clk)
                if (exp_mode && ph[6] && e10[g]) n_slot[g] <= int_y;           // n of exp E
        end
    endgenerate
    reg [15:0] r_m3, r_m4, r_m5, r_m6;
    reg [7:0]  n_ld;
    integer s;
    always @(*) begin
        r_m3 = 16'd0; r_m4 = 16'd0; r_m5 = 16'd0; r_m6 = 16'd0; n_ld = 8'd0;
        for (s = 0; s < 7; s = s + 1) begin
            r_m3 = r_m3 | ({16{e7[(s + 4) % 7]}} & r_slot[s]);   // exp E - 4
            r_m4 = r_m4 | ({16{e7[(s + 5) % 7]}} & r_slot[s]);   // exp E - 5
            r_m5 = r_m5 | ({16{e7[s]}}           & r_slot[s]);   // exp E - 7
            r_m6 = r_m6 | ({16{e7[(s + 1) % 7]}} & r_slot[s]);   // exp E - 8
        end
        for (s = 0; s < 10; s = s + 1)
            n_ld = n_ld | ({8{e10[s]}} & n_slot[s]);             // exp E - 10
    end

    // ---- unit operands: exp schedule by phase (AND-OR on the one-hot phase), or the
    // external operands
    wire [15:0] x_mul_a = ({16{ph[0]}} & exp_x) | ({16{ph[6]}} & sub_y) | ({16{ph[3]}} & C5) |
                          ({16{ph[2]}} & p_hold) | ({16{ph[5] | ph[1] | ph[4]}} & add_y);
    wire [15:0] x_mul_b = ({16{ph[0]}} & LOG2E) | ({16{ph[6]}} & LN2) | ({16{ph[3]}} & mul_y) |
                          ({16{ph[2]}} & r_m3) | ({16{ph[5]}} & r_m4) | ({16{ph[1]}} & r_m5) |
                          ({16{ph[4]}} & r_m6);
    wire [15:0] x_add_b = ({16{ph[0]}} & C4) | ({16{ph[6]}} & C3) | ({16{ph[2]}} & C2) |
                          ({16{ph[5]}} & C1) | ({16{ph[1] | ph[3] | ph[4]}} & C0);

    // operand registers of the multiplier, adder, subtractor and FP16 -> int8
    reg [15:0] q_mul_a, q_mul_b, q_add_a, q_add_b, q_sub_a, q_sub_b, q_int_a;
    always @(posedge clk) begin
        q_mul_a <= exp_mode ? x_mul_a : mul_a;
        q_mul_b <= exp_mode ? x_mul_b : mul_b;
        q_add_a <= exp_mode ? mul_y   : add_a;
        q_add_b <= exp_mode ? x_add_b : add_b;
        q_sub_a <= exp_mode ? t_hold  : sub_a;
        q_sub_b <= exp_mode ? i2f_y   : sub_b;
        q_int_a <= exp_mode ? mul_y   : int_a;
    end
    wire [7:0]  u_i2f_n   = exp_mode ? int_y : i2f_n;
    wire [15:0] u_ldexp_a = exp_mode ? add_y : ldexp_a;
    wire [7:0]  u_ldexp_n = exp_mode ? n_ld  : ldexp_n;

    wire [15:0] max_w, sub_w, add_w, div_w, mul_w, sqrt_w, i2f_w, ldexp_w;
    wire [7:0]  int_w;

    sfu_fp16_max     u_max   (.a(max_a), .b(max_b), .y(max_w));
    sfu_fp16_add     u_sub   (.clk(clk), .a(q_sub_a), .b(q_sub_b), .sub(1'b1), .y(sub_w));
    sfu_fp16_add     u_add   (.clk(clk), .a(q_add_a), .b(q_add_b), .sub(1'b0), .y(add_w));
    sfu_fp16_div     u_div   (.clk(clk), .a(div_a), .b(div_b), .y(div_w));
    sfu_fp16_mul     u_mul   (.clk(clk), .a(q_mul_a), .b(q_mul_b), .y(mul_w));
    sfu_fp16_sqrt    u_sqrt  (.clk(clk), .a(sqrt_a), .y(sqrt_w));
    sfu_fp16_to_int8 u_int   (.a(q_int_a), .y(int_w));
    sfu_int8_to_fp16 u_i2f   (.n(u_i2f_n), .y(i2f_w));
    sfu_fp16_ldexp   u_ldexp (.a(u_ldexp_a), .n(u_ldexp_n), .y(ldexp_w));

    always @(posedge clk) begin
        max_y   <= max_w;
        sub_y   <= sub_w;
        add_y   <= add_w;
        div_y   <= div_w;
        mul_y   <= mul_w;
        sqrt_y  <= sqrt_w;
        int_y   <= int_w;
        i2f_y   <= i2f_w;
        ldexp_y <= ldexp_w;
    end
endmodule
