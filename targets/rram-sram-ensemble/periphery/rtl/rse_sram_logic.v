// The logic of the SRAM macro (Krishnan et al., TCAD 2022, Sec. III-B, III-D-2, Fig. 4):
// 16 output-stationary PEs, the idle / load / compute / finish FSM, and the input and
// output buffers. The SRAM array itself is outside (sram_addr / sram_rdata).
//
// Each read delivers one bit to each PE (16-bit word, 16 PEs). A weight takes W reads; one
// input bit is applied per weight (bit-serial, later bits run the macro again, and the
// result is post-processed off chip, Sec. III-D-2).
module rse_pe #(
    parameter W = 3,
    parameter ACC = 9
) (
    input  wire           clk,
    input  wire           wgt_en,
    input  wire           act_en,
    input  wire           add_en,
    input  wire           clear,      // 2:1 mux: start from zero instead of the accumulator
    input  wire           wbit,
    input  wire           x,
    output wire [ACC-1:0] acc
);
    reg [W-1:0]   w_r;
    reg           x_r;
    reg [ACC-1:0] acc_r;
    wire [ACC-1:0] w_ext = {{(ACC-W){w_r[W-1]}}, w_r};
    wire [ACC-1:0] prod  = x_r ? w_ext : {ACC{1'b0}};
    wire [ACC-1:0] base  = clear ? {ACC{1'b0}} : acc_r;
    always @(posedge clk) begin
        if (wgt_en) w_r <= {w_r[W-2:0], wbit};
        if (act_en) x_r <= x;
        if (add_en) acc_r <= base + prod;
    end
    assign acc = acc_r;
endmodule

module rse_sram_logic #(
    parameter W = 3,
    parameter ACC = 9,
    parameter NW = 42,
    parameter PES = 16,
    parameter INB = 64
) (
    input  wire               clk,
    input  wire               rst,
    input  wire               start,
    input  wire               in_we,
    input  wire [5:0]         in_addr,
    input  wire               in_bit,
    input  wire [PES-1:0]     sram_rdata,
    output wire [6:0]         sram_addr,
    output wire               sram_ren,
    output wire               done,
    output wire [PES*ACC-1:0] out_flat
);
    localparam S_IDLE = 2'd0, S_LOAD = 2'd1, S_COMPUTE = 2'd2, S_FINISH = 2'd3;

    reg [1:0]       state;
    reg [6:0]       addr;
    reg [1:0]       bitc;
    reg [5:0]       k;
    reg             first;
    reg [INB-1:0]   in_buf;
    reg [PES*ACC-1:0] out_buf;

    wire load    = (state == S_LOAD);
    wire compute = (state == S_COMPUTE);
    wire finish  = (state == S_FINISH);
    wire last_bit = (bitc == W - 1);
    wire [PES*ACC-1:0] acc_flat;

    always @(posedge clk) begin
        if (in_we) in_buf[in_addr] <= in_bit;
        if (finish) out_buf <= acc_flat;
        if (rst) state <= S_IDLE;
        else case (state)
            S_IDLE: if (start) begin
                state <= S_LOAD;
                addr  <= 7'd0;
                bitc  <= 2'd0;
                k     <= 6'd0;
                first <= 1'b1;
            end
            S_LOAD: begin
                addr <= addr + 7'd1;
                bitc <= bitc + 2'd1;
                if (last_bit) state <= S_COMPUTE;
            end
            S_COMPUTE: begin
                first <= 1'b0;
                bitc  <= 2'd0;
                k     <= k + 6'd1;
                state <= (k == NW - 1) ? S_FINISH : S_LOAD;
            end
            default: state <= S_IDLE;
        endcase
    end

    genvar p;
    generate
        for (p = 0; p < PES; p = p + 1) begin : pe
            rse_pe #(.W(W), .ACC(ACC)) u (
                .clk(clk), .wgt_en(load), .act_en(load && last_bit), .add_en(compute), .clear(first),
                .wbit(sram_rdata[p]), .x(in_buf[k]), .acc(acc_flat[p*ACC +: ACC]));
        end
    endgenerate

    assign sram_addr = addr;
    assign sram_ren  = load;
    assign done      = finish;
    assign out_flat  = out_buf;
endmodule
