// Shift-and-add datapaths of one 1F1C array with ISAAC's weight-flip encoding
// (Shafiee et al., ISCA 2016, Sec. III "Encoding to Reduce ADC Size"): shift_add.v
// plus the flip correction and the per-column flip bits.
//   A column whose weights are collectively large is stored flipped (each 2-bit cell
//   holds 3 - W), so its ADC code c is sum_i a_i * (3 - W_i); its real value is
//   3 * U - c, with U = sum_i a_i the code of the array's unit column (its cells at
//   level 1). flip_q holds one bit per column, loaded with the weights (load_flip).
//   Each cell's value v (CV bits) = flip ? 3 * U - c : c; an output's partial sum is
//   v_0 + v_1 * 4 + v_2 * 16 + v_3 * 64 (PART bits).
//   Inputs arrive MSB first, two's complement: on the first (sign) bit acc = -partial,
//   then acc = 2 * acc + partial (ACC bits).
//   The column codes (and the unit column's) stay in the ADC output registers for a
//   whole conversion window, so UNITS datapaths are shared over the outputs: in slot s,
//   datapath u updates output s * UNITS + u (OUTS / UNITS slots per input bit).
//   The accumulators are not here: they live in UNITS SRAM banks of OUTS / UNITS words
//   (NVSim, characterize_shift_add.py). In slot s, datapath u reads word s of bank u
//   (acc_in) and the result (acc_out) is written back to it.
module shift_add_flip #(
    parameter OUTS  = 32,
    parameter CELLS = 4,
    parameter CODE  = 8,
    parameter CV    = 9,
    parameter PART  = 16,
    parameter ACC   = 24,
    parameter UNITS = 4
) (
    input  wire                          clk,
    input  wire                          load_flip,
    input  wire [OUTS*CELLS-1:0]         flip_in,
    input  wire                          first,
    input  wire [$clog2(OUTS/UNITS)-1:0] slot,
    input  wire [OUTS*CELLS*CODE-1:0]    codes,
    input  wire [CODE-1:0]               ucode,
    input  wire [UNITS*ACC-1:0]          acc_in,
    output wire [UNITS*ACC-1:0]          acc_out
);
    reg [OUTS*CELLS-1:0] flip_q;
    always @(posedge clk)
        if (load_flip)
            flip_q <= flip_in;

    // 3 * U, shared by every datapath
    wire [CV-1:0] u3 = ({{(CV-CODE){1'b0}}, ucode} << 1) + {{(CV-CODE){1'b0}}, ucode};

    genvar u, k;
    generate
        for (u = 0; u < UNITS; u = u + 1) begin : datapath
            wire [PART-1:0] sum [0:CELLS];
            assign sum[0] = {PART{1'b0}};
            for (k = 0; k < CELLS; k = k + 1) begin : cell
                wire [CODE-1:0] c = codes[((slot * UNITS + u) * CELLS + k) * CODE +: CODE];
                wire            f = flip_q[(slot * UNITS + u) * CELLS + k];
                wire [CV-1:0]   cz = {{(CV-CODE){1'b0}}, c};
                wire [CV-1:0]   v = f ? u3 - cz : cz;
                assign sum[k + 1] = sum[k] + ({{(PART-CV){1'b0}}, v} << (2 * k));
            end
            wire signed [ACC-1:0] p = $signed({{(ACC-PART){1'b0}}, sum[CELLS]});
            wire signed [ACC-1:0] a = acc_in[u * ACC +: ACC];
            assign acc_out[u * ACC +: ACC] = first ? -p : (a <<< 1) + p;
        end
    endgenerate
endmodule
