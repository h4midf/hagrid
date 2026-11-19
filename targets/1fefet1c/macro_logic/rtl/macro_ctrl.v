// Input and phase control of one 1F1C array for one MAC (one input bit per row).
// The word-line drivers themselves are in the NVSim array area (one per row); this is
// what replaces the memory row decoder's select:
//   input latches  in_bits <= din on load: the current input bit of every row
//   sequencer      start begins one MAC: reset (1 cycle), multiply (1 cycle),
//                  accumulate (1 cycle), then the ADC conversion (CONV_CYCLES cycles,
//                  samp high in its first cycle) (Laleni et al. 2024 Sec. III-B, V;
//                  targets/1fefet1c/default/design.json clock and adc)
//   word lines     wl[i] = in_bits[i] during the multiply phase
module macro_ctrl #(parameter ROWS = 128, parameter CONV_CYCLES = 8) (
    input  wire            clk,
    input  wire            start,
    input  wire            load,
    input  wire [ROWS-1:0] din,
    output wire [ROWS-1:0] wl,
    output wire            rst,
    output wire            mult,
    output wire            acc,
    output wire            samp,
    output wire            busy
);
    localparam LAST = 3 + CONV_CYCLES;
    localparam TW = $clog2(LAST + 1);

    reg [ROWS-1:0] in_bits;
    reg [TW-1:0]   t;

    always @(posedge clk) begin
        if (load)
            in_bits <= din;
        if (t == 0)
            t <= start ? 1 : 0;
        else if (t == LAST)
            t <= 0;
        else
            t <= t + 1;
    end

    assign rst  = (t == 1);
    assign mult = (t == 2);
    assign acc  = (t == 3);
    assign samp = (t == 4);
    assign busy = (t != 0);
    assign wl   = in_bits & {ROWS{mult}};
endmodule
