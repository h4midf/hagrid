// GELU in its tanh form (GPT-2's gelu_new):
//   GELU_tanh(x) = 0.5*x*(1 + tanh(sqrt(2/pi)*(x + 0.044715*x^3))).
// The same circuit as gelu.v (erf form): a direct piecewise-linear fit over [-8, 8) in
// 256 uniform segments (segment width 0.0625 = 2^16 in Q12.20 units at W=32, 2^4 in Q8.8
// units at W=16), each segment storing the exact value at its left edge (OFFSET) and
// the exact delta to its right edge (SLOPE), so the error is zero at every breakpoint;
// saturates to 0 for x < -8 and to x for x >= 8. Only the table constants differ: they
// are samples of the tanh form, rounded at each width's own scale (the W=16 table is not
// a truncation of the W=32 one). The two forms differ by up to 4.7e-4 (at x = -2.70),
// more than gelu.v's own fit error (3.9e-4), so a tanh-GELU model needs this table.
//
// CMOS logic, not memory (as gelu.v): 256 hardwired constants per width, selected by a
// 256-way case and consumed by one multiply-add; generate-if picks the width.
module gelu_tanh #(parameter W = 32, parameter FRAC = 20) (
    input  wire signed [W-1:0] x,
    output wire signed [W-1:0] y
);
    generate
        if (W == 32 && FRAC == 20) begin : q12_20
            localparam signed [32:0] DOM_Q = 33'sd8388608; // 8 * 2^20
            localparam FRAC_SHIFT = 16;                    // segment width = 2^16 = 0.0625

            wire signed [32:0] sum = $signed({x[31], x}) + DOM_Q;
            wire [7:0]  idx   = sum[23:16];
            wire [15:0] intra = sum[15:0];

            reg signed [23:0] off;
            reg signed [23:0] slope;
            always @* begin
                case (idx)
            8'd0: begin off = 24'sd0; slope = 24'sd0; end
            8'd1: begin off = 24'sd0; slope = 24'sd0; end
            8'd2: begin off = 24'sd0; slope = 24'sd0; end
            8'd3: begin off = 24'sd0; slope = 24'sd0; end
            8'd4: begin off = 24'sd0; slope = 24'sd0; end
            8'd5: begin off = 24'sd0; slope = 24'sd0; end
            8'd6: begin off = 24'sd0; slope = 24'sd0; end
            8'd7: begin off = 24'sd0; slope = 24'sd0; end
            8'd8: begin off = 24'sd0; slope = 24'sd0; end
            8'd9: begin off = 24'sd0; slope = 24'sd0; end
            8'd10: begin off = 24'sd0; slope = 24'sd0; end
            8'd11: begin off = 24'sd0; slope = 24'sd0; end
            8'd12: begin off = 24'sd0; slope = 24'sd0; end
            8'd13: begin off = 24'sd0; slope = 24'sd0; end
            8'd14: begin off = 24'sd0; slope = 24'sd0; end
            8'd15: begin off = 24'sd0; slope = 24'sd0; end
            8'd16: begin off = 24'sd0; slope = 24'sd0; end
            8'd17: begin off = 24'sd0; slope = 24'sd0; end
            8'd18: begin off = 24'sd0; slope = 24'sd0; end
            8'd19: begin off = 24'sd0; slope = 24'sd0; end
            8'd20: begin off = 24'sd0; slope = 24'sd0; end
            8'd21: begin off = 24'sd0; slope = 24'sd0; end
            8'd22: begin off = 24'sd0; slope = 24'sd0; end
            8'd23: begin off = 24'sd0; slope = 24'sd0; end
            8'd24: begin off = 24'sd0; slope = 24'sd0; end
            8'd25: begin off = 24'sd0; slope = 24'sd0; end
            8'd26: begin off = 24'sd0; slope = 24'sd0; end
            8'd27: begin off = 24'sd0; slope = 24'sd0; end
            8'd28: begin off = 24'sd0; slope = 24'sd0; end
            8'd29: begin off = 24'sd0; slope = 24'sd0; end
            8'd30: begin off = 24'sd0; slope = 24'sd0; end
            8'd31: begin off = 24'sd0; slope = 24'sd0; end
            8'd32: begin off = 24'sd0; slope = 24'sd0; end
            8'd33: begin off = 24'sd0; slope = 24'sd0; end
            8'd34: begin off = 24'sd0; slope = 24'sd0; end
            8'd35: begin off = 24'sd0; slope = 24'sd0; end
            8'd36: begin off = 24'sd0; slope = 24'sd0; end
            8'd37: begin off = 24'sd0; slope = 24'sd0; end
            8'd38: begin off = 24'sd0; slope = 24'sd0; end
            8'd39: begin off = 24'sd0; slope = 24'sd0; end
            8'd40: begin off = 24'sd0; slope = 24'sd0; end
            8'd41: begin off = 24'sd0; slope = 24'sd0; end
            8'd42: begin off = 24'sd0; slope = 24'sd0; end
            8'd43: begin off = 24'sd0; slope = 24'sd0; end
            8'd44: begin off = 24'sd0; slope = 24'sd0; end
            8'd45: begin off = 24'sd0; slope = 24'sd0; end
            8'd46: begin off = 24'sd0; slope = 24'sd0; end
            8'd47: begin off = 24'sd0; slope = 24'sd0; end
            8'd48: begin off = 24'sd0; slope = 24'sd0; end
            8'd49: begin off = 24'sd0; slope = 24'sd0; end
            8'd50: begin off = -24'sd1; slope = 24'sd0; end
            8'd51: begin off = -24'sd1; slope = 24'sd0; end
            8'd52: begin off = -24'sd1; slope = -24'sd1; end
            8'd53: begin off = -24'sd2; slope = -24'sd1; end
            8'd54: begin off = -24'sd3; slope = -24'sd1; end
            8'd55: begin off = -24'sd4; slope = -24'sd2; end
            8'd56: begin off = -24'sd5; slope = -24'sd2; end
            8'd57: begin off = -24'sd8; slope = -24'sd3; end
            8'd58: begin off = -24'sd11; slope = -24'sd4; end
            8'd59: begin off = -24'sd15; slope = -24'sd6; end
            8'd60: begin off = -24'sd21; slope = -24'sd8; end
            8'd61: begin off = -24'sd29; slope = -24'sd11; end
            8'd62: begin off = -24'sd40; slope = -24'sd14; end
            8'd63: begin off = -24'sd54; slope = -24'sd19; end
            8'd64: begin off = -24'sd74; slope = -24'sd25; end
            8'd65: begin off = -24'sd99; slope = -24'sd33; end
            8'd66: begin off = -24'sd132; slope = -24'sd43; end
            8'd67: begin off = -24'sd175; slope = -24'sd55; end
            8'd68: begin off = -24'sd230; slope = -24'sd71; end
            8'd69: begin off = -24'sd301; slope = -24'sd90; end
            8'd70: begin off = -24'sd390; slope = -24'sd113; end
            8'd71: begin off = -24'sd504; slope = -24'sd142; end
            8'd72: begin off = -24'sd646; slope = -24'sd177; end
            8'd73: begin off = -24'sd824; slope = -24'sd220; end
            8'd74: begin off = -24'sd1043; slope = -24'sd271; end
            8'd75: begin off = -24'sd1314; slope = -24'sd331; end
            8'd76: begin off = -24'sd1645; slope = -24'sd402; end
            8'd77: begin off = -24'sd2047; slope = -24'sd486; end
            8'd78: begin off = -24'sd2533; slope = -24'sd584; end
            8'd79: begin off = -24'sd3117; slope = -24'sd697; end
            8'd80: begin off = -24'sd3814; slope = -24'sd827; end
            8'd81: begin off = -24'sd4641; slope = -24'sd976; end
            8'd82: begin off = -24'sd5618; slope = -24'sd1145; end
            8'd83: begin off = -24'sd6763; slope = -24'sd1336; end
            8'd84: begin off = -24'sd8098; slope = -24'sd1549; end
            8'd85: begin off = -24'sd9647; slope = -24'sd1786; end
            8'd86: begin off = -24'sd11434; slope = -24'sd2048; end
            8'd87: begin off = -24'sd13482; slope = -24'sd2335; end
            8'd88: begin off = -24'sd15817; slope = -24'sd2648; end
            8'd89: begin off = -24'sd18465; slope = -24'sd2985; end
            8'd90: begin off = -24'sd21449; slope = -24'sd3345; end
            8'd91: begin off = -24'sd24794; slope = -24'sd3728; end
            8'd92: begin off = -24'sd28523; slope = -24'sd4131; end
            8'd93: begin off = -24'sd32654; slope = -24'sd4550; end
            8'd94: begin off = -24'sd37204; slope = -24'sd4982; end
            8'd95: begin off = -24'sd42186; slope = -24'sd5422; end
            8'd96: begin off = -24'sd47608; slope = -24'sd5863; end
            8'd97: begin off = -24'sd53471; slope = -24'sd6299; end
            8'd98: begin off = -24'sd59770; slope = -24'sd6722; end
            8'd99: begin off = -24'sd66492; slope = -24'sd7123; end
            8'd100: begin off = -24'sd73615; slope = -24'sd7492; end
            8'd101: begin off = -24'sd81107; slope = -24'sd7818; end
            8'd102: begin off = -24'sd88925; slope = -24'sd8089; end
            8'd103: begin off = -24'sd97014; slope = -24'sd8293; end
            8'd104: begin off = -24'sd105307; slope = -24'sd8417; end
            8'd105: begin off = -24'sd113724; slope = -24'sd8447; end
            8'd106: begin off = -24'sd122171; slope = -24'sd8370; end
            8'd107: begin off = -24'sd130541; slope = -24'sd8171; end
            8'd108: begin off = -24'sd138712; slope = -24'sd7837; end
            8'd109: begin off = -24'sd146549; slope = -24'sd7356; end
            8'd110: begin off = -24'sd153905; slope = -24'sd6715; end
            8'd111: begin off = -24'sd160620; slope = -24'sd5903; end
            8'd112: begin off = -24'sd166522; slope = -24'sd4910; end
            8'd113: begin off = -24'sd171432; slope = -24'sd3729; end
            8'd114: begin off = -24'sd175162; slope = -24'sd2355; end
            8'd115: begin off = -24'sd177517; slope = -24'sd783; end
            8'd116: begin off = -24'sd178299; slope = 24'sd988; end
            8'd117: begin off = -24'sd177311; slope = 24'sd2955; end
            8'd118: begin off = -24'sd174356; slope = 24'sd5115; end
            8'd119: begin off = -24'sd169241; slope = 24'sd7460; end
            8'd120: begin off = -24'sd161781; slope = 24'sd9980; end
            8'd121: begin off = -24'sd151800; slope = 24'sd12662; end
            8'd122: begin off = -24'sd139138; slope = 24'sd15491; end
            8'd123: begin off = -24'sd123647; slope = 24'sd18449; end
            8'd124: begin off = -24'sd105198; slope = 24'sd21514; end
            8'd125: begin off = -24'sd83684; slope = 24'sd24667; end
            8'd126: begin off = -24'sd59017; slope = 24'sd27882; end
            8'd127: begin off = -24'sd31135; slope = 24'sd31135; end
            8'd128: begin off = 24'sd0; slope = 24'sd34401; end
            8'd129: begin off = 24'sd34401; slope = 24'sd37654; end
            8'd130: begin off = 24'sd72055; slope = 24'sd40869; end
            8'd131: begin off = 24'sd112924; slope = 24'sd44022; end
            8'd132: begin off = 24'sd156946; slope = 24'sd47087; end
            8'd133: begin off = 24'sd204033; slope = 24'sd50045; end
            8'd134: begin off = 24'sd254078; slope = 24'sd52874; end
            8'd135: begin off = 24'sd306952; slope = 24'sd55556; end
            8'd136: begin off = 24'sd362507; slope = 24'sd58076; end
            8'd137: begin off = 24'sd420583; slope = 24'sd60421; end
            8'd138: begin off = 24'sd481004; slope = 24'sd62581; end
            8'd139: begin off = 24'sd543585; slope = 24'sd64548; end
            8'd140: begin off = 24'sd608133; slope = 24'sd66319; end
            8'd141: begin off = 24'sd674451; slope = 24'sd67891; end
            8'd142: begin off = 24'sd742342; slope = 24'sd69265; end
            8'd143: begin off = 24'sd811608; slope = 24'sd70446; end
            8'd144: begin off = 24'sd882054; slope = 24'sd71439; end
            8'd145: begin off = 24'sd953492; slope = 24'sd72251; end
            8'd146: begin off = 24'sd1025743; slope = 24'sd72892; end
            8'd147: begin off = 24'sd1098635; slope = 24'sd73373; end
            8'd148: begin off = 24'sd1172008; slope = 24'sd73707; end
            8'd149: begin off = 24'sd1245715; slope = 24'sd73906; end
            8'd150: begin off = 24'sd1319621; slope = 24'sd73983; end
            8'd151: begin off = 24'sd1393604; slope = 24'sd73953; end
            8'd152: begin off = 24'sd1467557; slope = 24'sd73829; end
            8'd153: begin off = 24'sd1541386; slope = 24'sd73625; end
            8'd154: begin off = 24'sd1615011; slope = 24'sd73354; end
            8'd155: begin off = 24'sd1688365; slope = 24'sd73028; end
            8'd156: begin off = 24'sd1761393; slope = 24'sd72659; end
            8'd157: begin off = 24'sd1834052; slope = 24'sd72258; end
            8'd158: begin off = 24'sd1906310; slope = 24'sd71835; end
            8'd159: begin off = 24'sd1978145; slope = 24'sd71399; end
            8'd160: begin off = 24'sd2049544; slope = 24'sd70958; end
            8'd161: begin off = 24'sd2120502; slope = 24'sd70518; end
            8'd162: begin off = 24'sd2191020; slope = 24'sd70086; end
            8'd163: begin off = 24'sd2261106; slope = 24'sd69667; end
            8'd164: begin off = 24'sd2330773; slope = 24'sd69264; end
            8'd165: begin off = 24'sd2400038; slope = 24'sd68881; end
            8'd166: begin off = 24'sd2468919; slope = 24'sd68521; end
            8'd167: begin off = 24'sd2537439; slope = 24'sd68184; end
            8'd168: begin off = 24'sd2605623; slope = 24'sd67871; end
            8'd169: begin off = 24'sd2673494; slope = 24'sd67584; end
            8'd170: begin off = 24'sd2741078; slope = 24'sd67322; end
            8'd171: begin off = 24'sd2808401; slope = 24'sd67085; end
            8'd172: begin off = 24'sd2875486; slope = 24'sd66872; end
            8'd173: begin off = 24'sd2942357; slope = 24'sd66681; end
            8'd174: begin off = 24'sd3009038; slope = 24'sd66512; end
            8'd175: begin off = 24'sd3075551; slope = 24'sd66363; end
            8'd176: begin off = 24'sd3141914; slope = 24'sd66233; end
            8'd177: begin off = 24'sd3208147; slope = 24'sd66120; end
            8'd178: begin off = 24'sd3274267; slope = 24'sd66022; end
            8'd179: begin off = 24'sd3340289; slope = 24'sd65938; end
            8'd180: begin off = 24'sd3406227; slope = 24'sd65867; end
            8'd181: begin off = 24'sd3472094; slope = 24'sd65807; end
            8'd182: begin off = 24'sd3537901; slope = 24'sd65756; end
            8'd183: begin off = 24'sd3603656; slope = 24'sd65713; end
            8'd184: begin off = 24'sd3669370; slope = 24'sd65678; end
            8'd185: begin off = 24'sd3735048; slope = 24'sd65649; end
            8'd186: begin off = 24'sd3800698; slope = 24'sd65626; end
            8'd187: begin off = 24'sd3866323; slope = 24'sd65607; end
            8'd188: begin off = 24'sd3931930; slope = 24'sd65591; end
            8'd189: begin off = 24'sd3997521; slope = 24'sd65579; end
            8'd190: begin off = 24'sd4063100; slope = 24'sd65569; end
            8'd191: begin off = 24'sd4128669; slope = 24'sd65561; end
            8'd192: begin off = 24'sd4194230; slope = 24'sd65555; end
            8'd193: begin off = 24'sd4259786; slope = 24'sd65550; end
            8'd194: begin off = 24'sd4325336; slope = 24'sd65547; end
            8'd195: begin off = 24'sd4390883; slope = 24'sd65544; end
            8'd196: begin off = 24'sd4456427; slope = 24'sd65542; end
            8'd197: begin off = 24'sd4521969; slope = 24'sd65540; end
            8'd198: begin off = 24'sd4587509; slope = 24'sd65539; end
            8'd199: begin off = 24'sd4653048; slope = 24'sd65538; end
            8'd200: begin off = 24'sd4718587; slope = 24'sd65538; end
            8'd201: begin off = 24'sd4784124; slope = 24'sd65537; end
            8'd202: begin off = 24'sd4849661; slope = 24'sd65537; end
            8'd203: begin off = 24'sd4915198; slope = 24'sd65537; end
            8'd204: begin off = 24'sd4980735; slope = 24'sd65536; end
            8'd205: begin off = 24'sd5046271; slope = 24'sd65536; end
            8'd206: begin off = 24'sd5111807; slope = 24'sd65536; end
            8'd207: begin off = 24'sd5177344; slope = 24'sd65536; end
            8'd208: begin off = 24'sd5242880; slope = 24'sd65536; end
            8'd209: begin off = 24'sd5308416; slope = 24'sd65536; end
            8'd210: begin off = 24'sd5373952; slope = 24'sd65536; end
            8'd211: begin off = 24'sd5439488; slope = 24'sd65536; end
            8'd212: begin off = 24'sd5505024; slope = 24'sd65536; end
            8'd213: begin off = 24'sd5570560; slope = 24'sd65536; end
            8'd214: begin off = 24'sd5636096; slope = 24'sd65536; end
            8'd215: begin off = 24'sd5701632; slope = 24'sd65536; end
            8'd216: begin off = 24'sd5767168; slope = 24'sd65536; end
            8'd217: begin off = 24'sd5832704; slope = 24'sd65536; end
            8'd218: begin off = 24'sd5898240; slope = 24'sd65536; end
            8'd219: begin off = 24'sd5963776; slope = 24'sd65536; end
            8'd220: begin off = 24'sd6029312; slope = 24'sd65536; end
            8'd221: begin off = 24'sd6094848; slope = 24'sd65536; end
            8'd222: begin off = 24'sd6160384; slope = 24'sd65536; end
            8'd223: begin off = 24'sd6225920; slope = 24'sd65536; end
            8'd224: begin off = 24'sd6291456; slope = 24'sd65536; end
            8'd225: begin off = 24'sd6356992; slope = 24'sd65536; end
            8'd226: begin off = 24'sd6422528; slope = 24'sd65536; end
            8'd227: begin off = 24'sd6488064; slope = 24'sd65536; end
            8'd228: begin off = 24'sd6553600; slope = 24'sd65536; end
            8'd229: begin off = 24'sd6619136; slope = 24'sd65536; end
            8'd230: begin off = 24'sd6684672; slope = 24'sd65536; end
            8'd231: begin off = 24'sd6750208; slope = 24'sd65536; end
            8'd232: begin off = 24'sd6815744; slope = 24'sd65536; end
            8'd233: begin off = 24'sd6881280; slope = 24'sd65536; end
            8'd234: begin off = 24'sd6946816; slope = 24'sd65536; end
            8'd235: begin off = 24'sd7012352; slope = 24'sd65536; end
            8'd236: begin off = 24'sd7077888; slope = 24'sd65536; end
            8'd237: begin off = 24'sd7143424; slope = 24'sd65536; end
            8'd238: begin off = 24'sd7208960; slope = 24'sd65536; end
            8'd239: begin off = 24'sd7274496; slope = 24'sd65536; end
            8'd240: begin off = 24'sd7340032; slope = 24'sd65536; end
            8'd241: begin off = 24'sd7405568; slope = 24'sd65536; end
            8'd242: begin off = 24'sd7471104; slope = 24'sd65536; end
            8'd243: begin off = 24'sd7536640; slope = 24'sd65536; end
            8'd244: begin off = 24'sd7602176; slope = 24'sd65536; end
            8'd245: begin off = 24'sd7667712; slope = 24'sd65536; end
            8'd246: begin off = 24'sd7733248; slope = 24'sd65536; end
            8'd247: begin off = 24'sd7798784; slope = 24'sd65536; end
            8'd248: begin off = 24'sd7864320; slope = 24'sd65536; end
            8'd249: begin off = 24'sd7929856; slope = 24'sd65536; end
            8'd250: begin off = 24'sd7995392; slope = 24'sd65536; end
            8'd251: begin off = 24'sd8060928; slope = 24'sd65536; end
            8'd252: begin off = 24'sd8126464; slope = 24'sd65536; end
            8'd253: begin off = 24'sd8192000; slope = 24'sd65536; end
            8'd254: begin off = 24'sd8257536; slope = 24'sd65536; end
            8'd255: begin off = 24'sd8323072; slope = 24'sd65536; end
                endcase
            end

            wire signed [40:0] prod   = slope * $signed({1'b0, intra});
            wire signed [31:0] interp = off + (prod >>> FRAC_SHIFT);

            wire below = x < -32'sd8388608;
            wire above = x >= 32'sd8388608;
            assign y = below ? 32'sd0 : above ? x : interp;
        end else if (W == 16 && FRAC == 8) begin : q8_8
            localparam signed [16:0] DOM_Q = 17'sd2048; // 8 * 2^8
            localparam FRAC_SHIFT = 4;                   // segment width = 2^4 = 0.0625

            wire signed [16:0] sum = $signed({x[15], x}) + DOM_Q;
            wire [7:0] idx   = sum[11:4];
            wire [3:0] intra = sum[3:0];

            reg signed [23:0] off;
            reg signed [23:0] slope;
            always @* begin
                case (idx)
            8'd0: begin off = 24'sd0; slope = 24'sd0; end
            8'd1: begin off = 24'sd0; slope = 24'sd0; end
            8'd2: begin off = 24'sd0; slope = 24'sd0; end
            8'd3: begin off = 24'sd0; slope = 24'sd0; end
            8'd4: begin off = 24'sd0; slope = 24'sd0; end
            8'd5: begin off = 24'sd0; slope = 24'sd0; end
            8'd6: begin off = 24'sd0; slope = 24'sd0; end
            8'd7: begin off = 24'sd0; slope = 24'sd0; end
            8'd8: begin off = 24'sd0; slope = 24'sd0; end
            8'd9: begin off = 24'sd0; slope = 24'sd0; end
            8'd10: begin off = 24'sd0; slope = 24'sd0; end
            8'd11: begin off = 24'sd0; slope = 24'sd0; end
            8'd12: begin off = 24'sd0; slope = 24'sd0; end
            8'd13: begin off = 24'sd0; slope = 24'sd0; end
            8'd14: begin off = 24'sd0; slope = 24'sd0; end
            8'd15: begin off = 24'sd0; slope = 24'sd0; end
            8'd16: begin off = 24'sd0; slope = 24'sd0; end
            8'd17: begin off = 24'sd0; slope = 24'sd0; end
            8'd18: begin off = 24'sd0; slope = 24'sd0; end
            8'd19: begin off = 24'sd0; slope = 24'sd0; end
            8'd20: begin off = 24'sd0; slope = 24'sd0; end
            8'd21: begin off = 24'sd0; slope = 24'sd0; end
            8'd22: begin off = 24'sd0; slope = 24'sd0; end
            8'd23: begin off = 24'sd0; slope = 24'sd0; end
            8'd24: begin off = 24'sd0; slope = 24'sd0; end
            8'd25: begin off = 24'sd0; slope = 24'sd0; end
            8'd26: begin off = 24'sd0; slope = 24'sd0; end
            8'd27: begin off = 24'sd0; slope = 24'sd0; end
            8'd28: begin off = 24'sd0; slope = 24'sd0; end
            8'd29: begin off = 24'sd0; slope = 24'sd0; end
            8'd30: begin off = 24'sd0; slope = 24'sd0; end
            8'd31: begin off = 24'sd0; slope = 24'sd0; end
            8'd32: begin off = 24'sd0; slope = 24'sd0; end
            8'd33: begin off = 24'sd0; slope = 24'sd0; end
            8'd34: begin off = 24'sd0; slope = 24'sd0; end
            8'd35: begin off = 24'sd0; slope = 24'sd0; end
            8'd36: begin off = 24'sd0; slope = 24'sd0; end
            8'd37: begin off = 24'sd0; slope = 24'sd0; end
            8'd38: begin off = 24'sd0; slope = 24'sd0; end
            8'd39: begin off = 24'sd0; slope = 24'sd0; end
            8'd40: begin off = 24'sd0; slope = 24'sd0; end
            8'd41: begin off = 24'sd0; slope = 24'sd0; end
            8'd42: begin off = 24'sd0; slope = 24'sd0; end
            8'd43: begin off = 24'sd0; slope = 24'sd0; end
            8'd44: begin off = 24'sd0; slope = 24'sd0; end
            8'd45: begin off = 24'sd0; slope = 24'sd0; end
            8'd46: begin off = 24'sd0; slope = 24'sd0; end
            8'd47: begin off = 24'sd0; slope = 24'sd0; end
            8'd48: begin off = 24'sd0; slope = 24'sd0; end
            8'd49: begin off = 24'sd0; slope = 24'sd0; end
            8'd50: begin off = 24'sd0; slope = 24'sd0; end
            8'd51: begin off = 24'sd0; slope = 24'sd0; end
            8'd52: begin off = 24'sd0; slope = 24'sd0; end
            8'd53: begin off = 24'sd0; slope = 24'sd0; end
            8'd54: begin off = 24'sd0; slope = 24'sd0; end
            8'd55: begin off = 24'sd0; slope = 24'sd0; end
            8'd56: begin off = 24'sd0; slope = 24'sd0; end
            8'd57: begin off = 24'sd0; slope = 24'sd0; end
            8'd58: begin off = 24'sd0; slope = 24'sd0; end
            8'd59: begin off = 24'sd0; slope = 24'sd0; end
            8'd60: begin off = 24'sd0; slope = 24'sd0; end
            8'd61: begin off = 24'sd0; slope = 24'sd0; end
            8'd62: begin off = 24'sd0; slope = 24'sd0; end
            8'd63: begin off = 24'sd0; slope = 24'sd0; end
            8'd64: begin off = 24'sd0; slope = 24'sd0; end
            8'd65: begin off = 24'sd0; slope = 24'sd0; end
            8'd66: begin off = 24'sd0; slope = 24'sd0; end
            8'd67: begin off = 24'sd0; slope = 24'sd0; end
            8'd68: begin off = 24'sd0; slope = 24'sd0; end
            8'd69: begin off = 24'sd0; slope = 24'sd0; end
            8'd70: begin off = 24'sd0; slope = 24'sd0; end
            8'd71: begin off = 24'sd0; slope = 24'sd0; end
            8'd72: begin off = 24'sd0; slope = 24'sd0; end
            8'd73: begin off = 24'sd0; slope = 24'sd0; end
            8'd74: begin off = 24'sd0; slope = 24'sd0; end
            8'd75: begin off = 24'sd0; slope = 24'sd0; end
            8'd76: begin off = 24'sd0; slope = 24'sd0; end
            8'd77: begin off = 24'sd0; slope = 24'sd0; end
            8'd78: begin off = -24'sd1; slope = 24'sd0; end
            8'd79: begin off = -24'sd1; slope = 24'sd0; end
            8'd80: begin off = -24'sd1; slope = 24'sd0; end
            8'd81: begin off = -24'sd1; slope = 24'sd0; end
            8'd82: begin off = -24'sd1; slope = 24'sd0; end
            8'd83: begin off = -24'sd2; slope = 24'sd0; end
            8'd84: begin off = -24'sd2; slope = 24'sd0; end
            8'd85: begin off = -24'sd2; slope = 24'sd0; end
            8'd86: begin off = -24'sd3; slope = -24'sd1; end
            8'd87: begin off = -24'sd3; slope = -24'sd1; end
            8'd88: begin off = -24'sd4; slope = -24'sd1; end
            8'd89: begin off = -24'sd5; slope = -24'sd1; end
            8'd90: begin off = -24'sd5; slope = -24'sd1; end
            8'd91: begin off = -24'sd6; slope = -24'sd1; end
            8'd92: begin off = -24'sd7; slope = -24'sd1; end
            8'd93: begin off = -24'sd8; slope = -24'sd1; end
            8'd94: begin off = -24'sd9; slope = -24'sd1; end
            8'd95: begin off = -24'sd10; slope = -24'sd1; end
            8'd96: begin off = -24'sd12; slope = -24'sd1; end
            8'd97: begin off = -24'sd13; slope = -24'sd2; end
            8'd98: begin off = -24'sd15; slope = -24'sd2; end
            8'd99: begin off = -24'sd16; slope = -24'sd2; end
            8'd100: begin off = -24'sd18; slope = -24'sd2; end
            8'd101: begin off = -24'sd20; slope = -24'sd2; end
            8'd102: begin off = -24'sd22; slope = -24'sd2; end
            8'd103: begin off = -24'sd24; slope = -24'sd2; end
            8'd104: begin off = -24'sd26; slope = -24'sd2; end
            8'd105: begin off = -24'sd28; slope = -24'sd2; end
            8'd106: begin off = -24'sd30; slope = -24'sd2; end
            8'd107: begin off = -24'sd32; slope = -24'sd2; end
            8'd108: begin off = -24'sd34; slope = -24'sd2; end
            8'd109: begin off = -24'sd36; slope = -24'sd2; end
            8'd110: begin off = -24'sd38; slope = -24'sd2; end
            8'd111: begin off = -24'sd39; slope = -24'sd1; end
            8'd112: begin off = -24'sd41; slope = -24'sd1; end
            8'd113: begin off = -24'sd42; slope = -24'sd1; end
            8'd114: begin off = -24'sd43; slope = -24'sd1; end
            8'd115: begin off = -24'sd43; slope = 24'sd0; end
            8'd116: begin off = -24'sd44; slope = 24'sd0; end
            8'd117: begin off = -24'sd43; slope = 24'sd1; end
            8'd118: begin off = -24'sd43; slope = 24'sd1; end
            8'd119: begin off = -24'sd41; slope = 24'sd2; end
            8'd120: begin off = -24'sd39; slope = 24'sd2; end
            8'd121: begin off = -24'sd37; slope = 24'sd3; end
            8'd122: begin off = -24'sd34; slope = 24'sd4; end
            8'd123: begin off = -24'sd30; slope = 24'sd5; end
            8'd124: begin off = -24'sd26; slope = 24'sd5; end
            8'd125: begin off = -24'sd20; slope = 24'sd6; end
            8'd126: begin off = -24'sd14; slope = 24'sd7; end
            8'd127: begin off = -24'sd8; slope = 24'sd8; end
            8'd128: begin off = 24'sd0; slope = 24'sd8; end
            8'd129: begin off = 24'sd8; slope = 24'sd9; end
            8'd130: begin off = 24'sd18; slope = 24'sd10; end
            8'd131: begin off = 24'sd28; slope = 24'sd11; end
            8'd132: begin off = 24'sd38; slope = 24'sd11; end
            8'd133: begin off = 24'sd50; slope = 24'sd12; end
            8'd134: begin off = 24'sd62; slope = 24'sd13; end
            8'd135: begin off = 24'sd75; slope = 24'sd14; end
            8'd136: begin off = 24'sd89; slope = 24'sd14; end
            8'd137: begin off = 24'sd103; slope = 24'sd15; end
            8'd138: begin off = 24'sd117; slope = 24'sd15; end
            8'd139: begin off = 24'sd133; slope = 24'sd16; end
            8'd140: begin off = 24'sd148; slope = 24'sd16; end
            8'd141: begin off = 24'sd165; slope = 24'sd17; end
            8'd142: begin off = 24'sd181; slope = 24'sd17; end
            8'd143: begin off = 24'sd198; slope = 24'sd17; end
            8'd144: begin off = 24'sd215; slope = 24'sd17; end
            8'd145: begin off = 24'sd233; slope = 24'sd18; end
            8'd146: begin off = 24'sd250; slope = 24'sd18; end
            8'd147: begin off = 24'sd268; slope = 24'sd18; end
            8'd148: begin off = 24'sd286; slope = 24'sd18; end
            8'd149: begin off = 24'sd304; slope = 24'sd18; end
            8'd150: begin off = 24'sd322; slope = 24'sd18; end
            8'd151: begin off = 24'sd340; slope = 24'sd18; end
            8'd152: begin off = 24'sd358; slope = 24'sd18; end
            8'd153: begin off = 24'sd376; slope = 24'sd18; end
            8'd154: begin off = 24'sd394; slope = 24'sd18; end
            8'd155: begin off = 24'sd412; slope = 24'sd18; end
            8'd156: begin off = 24'sd430; slope = 24'sd18; end
            8'd157: begin off = 24'sd448; slope = 24'sd18; end
            8'd158: begin off = 24'sd465; slope = 24'sd18; end
            8'd159: begin off = 24'sd483; slope = 24'sd17; end
            8'd160: begin off = 24'sd500; slope = 24'sd17; end
            8'd161: begin off = 24'sd518; slope = 24'sd17; end
            8'd162: begin off = 24'sd535; slope = 24'sd17; end
            8'd163: begin off = 24'sd552; slope = 24'sd17; end
            8'd164: begin off = 24'sd569; slope = 24'sd17; end
            8'd165: begin off = 24'sd586; slope = 24'sd17; end
            8'd166: begin off = 24'sd603; slope = 24'sd17; end
            8'd167: begin off = 24'sd619; slope = 24'sd17; end
            8'd168: begin off = 24'sd636; slope = 24'sd17; end
            8'd169: begin off = 24'sd653; slope = 24'sd17; end
            8'd170: begin off = 24'sd669; slope = 24'sd16; end
            8'd171: begin off = 24'sd686; slope = 24'sd16; end
            8'd172: begin off = 24'sd702; slope = 24'sd16; end
            8'd173: begin off = 24'sd718; slope = 24'sd16; end
            8'd174: begin off = 24'sd735; slope = 24'sd16; end
            8'd175: begin off = 24'sd751; slope = 24'sd16; end
            8'd176: begin off = 24'sd767; slope = 24'sd16; end
            8'd177: begin off = 24'sd783; slope = 24'sd16; end
            8'd178: begin off = 24'sd799; slope = 24'sd16; end
            8'd179: begin off = 24'sd816; slope = 24'sd16; end
            8'd180: begin off = 24'sd832; slope = 24'sd16; end
            8'd181: begin off = 24'sd848; slope = 24'sd16; end
            8'd182: begin off = 24'sd864; slope = 24'sd16; end
            8'd183: begin off = 24'sd880; slope = 24'sd16; end
            8'd184: begin off = 24'sd896; slope = 24'sd16; end
            8'd185: begin off = 24'sd912; slope = 24'sd16; end
            8'd186: begin off = 24'sd928; slope = 24'sd16; end
            8'd187: begin off = 24'sd944; slope = 24'sd16; end
            8'd188: begin off = 24'sd960; slope = 24'sd16; end
            8'd189: begin off = 24'sd976; slope = 24'sd16; end
            8'd190: begin off = 24'sd992; slope = 24'sd16; end
            8'd191: begin off = 24'sd1008; slope = 24'sd16; end
            8'd192: begin off = 24'sd1024; slope = 24'sd16; end
            8'd193: begin off = 24'sd1040; slope = 24'sd16; end
            8'd194: begin off = 24'sd1056; slope = 24'sd16; end
            8'd195: begin off = 24'sd1072; slope = 24'sd16; end
            8'd196: begin off = 24'sd1088; slope = 24'sd16; end
            8'd197: begin off = 24'sd1104; slope = 24'sd16; end
            8'd198: begin off = 24'sd1120; slope = 24'sd16; end
            8'd199: begin off = 24'sd1136; slope = 24'sd16; end
            8'd200: begin off = 24'sd1152; slope = 24'sd16; end
            8'd201: begin off = 24'sd1168; slope = 24'sd16; end
            8'd202: begin off = 24'sd1184; slope = 24'sd16; end
            8'd203: begin off = 24'sd1200; slope = 24'sd16; end
            8'd204: begin off = 24'sd1216; slope = 24'sd16; end
            8'd205: begin off = 24'sd1232; slope = 24'sd16; end
            8'd206: begin off = 24'sd1248; slope = 24'sd16; end
            8'd207: begin off = 24'sd1264; slope = 24'sd16; end
            8'd208: begin off = 24'sd1280; slope = 24'sd16; end
            8'd209: begin off = 24'sd1296; slope = 24'sd16; end
            8'd210: begin off = 24'sd1312; slope = 24'sd16; end
            8'd211: begin off = 24'sd1328; slope = 24'sd16; end
            8'd212: begin off = 24'sd1344; slope = 24'sd16; end
            8'd213: begin off = 24'sd1360; slope = 24'sd16; end
            8'd214: begin off = 24'sd1376; slope = 24'sd16; end
            8'd215: begin off = 24'sd1392; slope = 24'sd16; end
            8'd216: begin off = 24'sd1408; slope = 24'sd16; end
            8'd217: begin off = 24'sd1424; slope = 24'sd16; end
            8'd218: begin off = 24'sd1440; slope = 24'sd16; end
            8'd219: begin off = 24'sd1456; slope = 24'sd16; end
            8'd220: begin off = 24'sd1472; slope = 24'sd16; end
            8'd221: begin off = 24'sd1488; slope = 24'sd16; end
            8'd222: begin off = 24'sd1504; slope = 24'sd16; end
            8'd223: begin off = 24'sd1520; slope = 24'sd16; end
            8'd224: begin off = 24'sd1536; slope = 24'sd16; end
            8'd225: begin off = 24'sd1552; slope = 24'sd16; end
            8'd226: begin off = 24'sd1568; slope = 24'sd16; end
            8'd227: begin off = 24'sd1584; slope = 24'sd16; end
            8'd228: begin off = 24'sd1600; slope = 24'sd16; end
            8'd229: begin off = 24'sd1616; slope = 24'sd16; end
            8'd230: begin off = 24'sd1632; slope = 24'sd16; end
            8'd231: begin off = 24'sd1648; slope = 24'sd16; end
            8'd232: begin off = 24'sd1664; slope = 24'sd16; end
            8'd233: begin off = 24'sd1680; slope = 24'sd16; end
            8'd234: begin off = 24'sd1696; slope = 24'sd16; end
            8'd235: begin off = 24'sd1712; slope = 24'sd16; end
            8'd236: begin off = 24'sd1728; slope = 24'sd16; end
            8'd237: begin off = 24'sd1744; slope = 24'sd16; end
            8'd238: begin off = 24'sd1760; slope = 24'sd16; end
            8'd239: begin off = 24'sd1776; slope = 24'sd16; end
            8'd240: begin off = 24'sd1792; slope = 24'sd16; end
            8'd241: begin off = 24'sd1808; slope = 24'sd16; end
            8'd242: begin off = 24'sd1824; slope = 24'sd16; end
            8'd243: begin off = 24'sd1840; slope = 24'sd16; end
            8'd244: begin off = 24'sd1856; slope = 24'sd16; end
            8'd245: begin off = 24'sd1872; slope = 24'sd16; end
            8'd246: begin off = 24'sd1888; slope = 24'sd16; end
            8'd247: begin off = 24'sd1904; slope = 24'sd16; end
            8'd248: begin off = 24'sd1920; slope = 24'sd16; end
            8'd249: begin off = 24'sd1936; slope = 24'sd16; end
            8'd250: begin off = 24'sd1952; slope = 24'sd16; end
            8'd251: begin off = 24'sd1968; slope = 24'sd16; end
            8'd252: begin off = 24'sd1984; slope = 24'sd16; end
            8'd253: begin off = 24'sd2000; slope = 24'sd16; end
            8'd254: begin off = 24'sd2016; slope = 24'sd16; end
            8'd255: begin off = 24'sd2032; slope = 24'sd16; end
                endcase
            end

            wire signed [27:0] prod   = slope * $signed({1'b0, intra});
            wire signed [15:0] interp = off + (prod >>> FRAC_SHIFT);

            wire below = x < -16'sd2048;
            wire above = x >= 16'sd2048;
            assign y = below ? 16'sd0 : above ? x : interp;
        end
    endgenerate
endmodule
