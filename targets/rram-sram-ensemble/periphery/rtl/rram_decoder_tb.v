// Stimulus for test_rram_decoder.py. Each cycle shifts {state, adc_en, done, col_sel} into
// trace; sel_cap holds the WL, BL and SL selects seen in the write state.
module rram_decoder_tb (
    input  wire              clk,
    output reg [12*24-1:0]   trace,
    output reg [191:0]       sel_cap
);
    initial begin
        trace = 0;
        sel_cap = 0;
    end
    reg [7:0] t = 0;
    wire rst       = (t < 2);
    wire go        = (t == 3);
    wire write_req = (t >= 15) && (t < 18);

    wire [1:0]  state;
    wire [7:0]  col_sel;
    wire        adc_en, done;
    wire [63:0] wl, bl, sl;
    rram_decoder dut (.clk(clk), .rst(rst), .go(go), .write_req(write_req), .row(6'd37), .col(6'd5),
                      .state_o(state), .col_sel(col_sel), .adc_en(adc_en), .done(done),
                      .wl_sel(wl), .bl_sel(bl), .sl_sel(sl));

    always @(posedge clk) begin
        t <= t + 8'd1;
        if (t < 24) trace <= {trace[12*23-1:0], state, adc_en, done, col_sel};
        if (state == 2'd2) sel_cap <= {wl, bl, sl};
    end
endmodule
