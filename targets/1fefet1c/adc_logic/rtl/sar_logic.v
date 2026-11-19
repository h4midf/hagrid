// Digital part of the 7-bit SAR ADC of one 1F1C column (Laleni et al. 2024, Fig. 9),
// written from the SPICE netlist targets/1fefet1c/spice/adc/sar_core.lib (sarbit,
// dacsw, dff) and sar_core_tb.sp (shift register, sample/clock inverters):
//   shift register  t[6] <= samp, t[k-1] <= t[k]: the one-hot trial bit of each cycle
//   bit registers   b[k] <= samp ? 0 : (t[k] ? cmp : b[k])
//   DAC control     d[k] = t[k] | b[k]; enref = ~samp & d (bottom plate to Vref),
//                   engnd = ~samp & ~d (bottom plate to ground)
//   inverters       clkc = ~clk (comparator clock), sampb = ~samp (sampling switches)
// The analog parts (CDAC, its switches, comparator, buffer, sampling switch) are not
// in this module.
module sar_logic (
    input  wire       clk,
    input  wire       samp,
    input  wire       cmp,
    output reg  [6:0] b,
    output wire [6:0] enref,
    output wire [6:0] engnd,
    output wire       clkc,
    output wire       sampb
);
    reg  [6:0] t;
    wire [6:0] d = t | b;

    always @(posedge clk) begin
        t <= {samp, t[6:1]};
        b <= samp ? 7'd0 : ((t & {7{cmp}}) | (~t & b));
    end

    assign enref = {7{~samp}} & d;
    assign engnd = {7{~samp}} & ~d;
    assign clkc  = ~clk;
    assign sampb = ~samp;
endmodule
