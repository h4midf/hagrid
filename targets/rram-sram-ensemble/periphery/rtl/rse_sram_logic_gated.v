// rse_sram_logic.v with clock gating: each register bank is clocked through one clock gate
// on its enable (Sec. III-D-2: 'PE control is generated to enable the D flip-flops'), so the
// flops need no enable multiplexers. Same ports, FSM and results as rse_sram_logic.v.
module rse_icg (
    input  wire clk,
    input  wire en,
    output wire gclk
);
    // latch the enable while the clock is low, so the gated clock cannot glitch
    reg en_l;
    always @(*) if (!clk) en_l = en;
    assign gclk = clk & en_l;
endmodule

module rse_pe #(
    parameter W = 3,
    parameter ACC = 9
) (
    input  wire           clk_w,      // gated by wgt_en
    input  wire           clk_acc,    // gated by add_en
    input  wire           clear,
    input  wire           wbit,
    input  wire           x_r,        // the latched input bit, shared by all PEs
    output wire [ACC-1:0] acc
);
    reg [W-1:0]   w_r;
    reg [ACC-1:0] acc_r;
    wire [ACC-1:0] w_ext = {{(ACC-W){w_r[W-1]}}, w_r};
    wire [ACC-1:0] prod  = x_r ? w_ext : {ACC{1'b0}};
    wire [ACC-1:0] base  = clear ? {ACC{1'b0}} : acc_r;
    always @(posedge clk_w)   w_r   <= {w_r[W-2:0], wbit};
    always @(posedge clk_acc) acc_r <= base + prod;
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

    reg [1:0]         state;
    reg [6:0]         addr;
    reg [1:0]         bitc;
    reg [5:0]         k;
    reg               first;
    reg [INB-1:0]     in_buf;
    reg               x_r;
    reg [PES*ACC-1:0] out_buf;

    wire load     = (state == S_LOAD);
    wire compute  = (state == S_COMPUTE);
    wire finish   = (state == S_FINISH);
    wire last_bit = (bitc == W - 1);
    wire [PES*ACC-1:0] acc_flat;

    wire clk_w, clk_x, clk_acc, clk_in, clk_out;
    rse_icg g_w   (.clk(clk), .en(load),             .gclk(clk_w));
    rse_icg g_x   (.clk(clk), .en(load && last_bit), .gclk(clk_x));
    rse_icg g_acc (.clk(clk), .en(compute),          .gclk(clk_acc));
    rse_icg g_in  (.clk(clk), .en(in_we),            .gclk(clk_in));
    rse_icg g_out (.clk(clk), .en(finish),           .gclk(clk_out));

    always @(posedge clk_in)  in_buf[in_addr] <= in_bit;
    always @(posedge clk_x)   x_r <= in_buf[k];
    always @(posedge clk_out) out_buf <= acc_flat;

    always @(posedge clk) begin
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
                .clk_w(clk_w), .clk_acc(clk_acc), .clear(first), .wbit(sram_rdata[p]), .x_r(x_r),
                .acc(acc_flat[p*ACC +: ACC]));
        end
    endgenerate

    assign sram_addr = addr;
    assign sram_ren  = load;
    assign done      = finish;
    assign out_flat  = out_buf;
endmodule
