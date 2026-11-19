// RRAM decoder of the RRAM macro (Krishnan et al., TCAD 2022, Sec. III-D-1): an FSM with
// write, compute and input states. Write programs one cell, chosen by one-hot WL, BL and
// SL selects. Compute steps the 8-to-1 column mux over 8 cycles with the ADCs enabled.
// Input waits for the next stream of input activations.
module rram_decoder (
    input  wire        clk,
    input  wire        rst,
    input  wire        go,          // inputs are on the WLs: run one MAC
    input  wire        write_req,
    input  wire [5:0]  row,
    input  wire [5:0]  col,
    output wire [1:0]  state_o,
    output wire [7:0]  col_sel,     // column mux, one-hot
    output wire        adc_en,
    output wire        done,
    output wire [63:0] wl_sel,      // programming selects, one-hot
    output wire [63:0] bl_sel,
    output wire [63:0] sl_sel
);
    localparam S_INPUT = 2'd0, S_COMPUTE = 2'd1, S_WRITE = 2'd2;

    reg [1:0] state;
    reg [2:0] cnt;

    always @(posedge clk)
        if (rst) begin
            state <= S_INPUT;
            cnt   <= 3'd0;
        end else
            case (state)
                S_INPUT:
                    if (write_req) state <= S_WRITE;
                    else if (go) begin
                        state <= S_COMPUTE;
                        cnt   <= 3'd0;
                    end
                S_COMPUTE: begin
                    cnt <= cnt + 3'd1;
                    if (cnt == 3'd7) state <= S_INPUT;
                end
                S_WRITE:
                    if (!write_req) state <= S_INPUT;
                default: state <= S_INPUT;
            endcase

    wire write = (state == S_WRITE);
    assign state_o = state;
    assign adc_en  = (state == S_COMPUTE);
    assign col_sel = adc_en ? (8'd1 << cnt) : 8'd0;
    assign done    = adc_en && (cnt == 3'd7);
    assign wl_sel  = write ? (64'd1 << row) : 64'd0;
    assign bl_sel  = write ? (64'd1 << col) : 64'd0;
    assign sl_sel  = write ? (64'd1 << col) : 64'd0;
endmodule
