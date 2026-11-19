// GELU: a direct piecewise-linear fit of GELU (not a tanh / exp decomposition, which
// would compound two approximations). [-8, 8) in 256 uniform segments (2^16 Q12.20 units
// at W=32, 2^4 Q8.8 units at W=16), so the segment select is one shift and one compare.
// Each segment stores the exact GELU at its left edge (OFFSET) and the delta to its right
// edge (SLOPE): zero error at every breakpoint. Saturates to 0 for x < -8 and to x for
// x >= 8.
//
// Tables from GELU(x) = 0.5*x*(1+erf(x/sqrt(2))), at each width's own scale (the Q8.8
// table is not a truncation of the Q12.20 one). Error vs. true GELU
// (approximation_error.py): Q12.20 mean 2.5e-5, max 3.9e-4; Q8.8 mean 1.0e-3, max 8.9e-3.
//
// CMOS logic, not memory: 256 hardwired constants per width, selected by a 256-way case
// into one multiply-add; generate-if picks the width.
module gelu #(parameter W = 32, parameter FRAC = 20) (
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
            8'd45: begin off = -24'sd1; slope = 24'sd0; end
            8'd46: begin off = -24'sd1; slope = 24'sd0; end
            8'd47: begin off = -24'sd1; slope = 24'sd0; end
            8'd48: begin off = -24'sd2; slope = -24'sd1; end
            8'd49: begin off = -24'sd2; slope = -24'sd1; end
            8'd50: begin off = -24'sd3; slope = -24'sd1; end
            8'd51: begin off = -24'sd4; slope = -24'sd1; end
            8'd52: begin off = -24'sd5; slope = -24'sd2; end
            8'd53: begin off = -24'sd7; slope = -24'sd2; end
            8'd54: begin off = -24'sd9; slope = -24'sd3; end
            8'd55: begin off = -24'sd12; slope = -24'sd4; end
            8'd56: begin off = -24'sd16; slope = -24'sd5; end
            8'd57: begin off = -24'sd21; slope = -24'sd7; end
            8'd58: begin off = -24'sd28; slope = -24'sd9; end
            8'd59: begin off = -24'sd36; slope = -24'sd11; end
            8'd60: begin off = -24'sd48; slope = -24'sd14; end
            8'd61: begin off = -24'sd62; slope = -24'sd18; end
            8'd62: begin off = -24'sd80; slope = -24'sd23; end
            8'd63: begin off = -24'sd103; slope = -24'sd29; end
            8'd64: begin off = -24'sd133; slope = -24'sd37; end
            8'd65: begin off = -24'sd170; slope = -24'sd47; end
            8'd66: begin off = -24'sd217; slope = -24'sd58; end
            8'd67: begin off = -24'sd275; slope = -24'sd73; end
            8'd68: begin off = -24'sd348; slope = -24'sd90; end
            8'd69: begin off = -24'sd438; slope = -24'sd111; end
            8'd70: begin off = -24'sd549; slope = -24'sd137; end
            8'd71: begin off = -24'sd686; slope = -24'sd168; end
            8'd72: begin off = -24'sd854; slope = -24'sd204; end
            8'd73: begin off = -24'sd1058; slope = -24'sd248; end
            8'd74: begin off = -24'sd1306; slope = -24'sd300; end
            8'd75: begin off = -24'sd1606; slope = -24'sd361; end
            8'd76: begin off = -24'sd1966; slope = -24'sd432; end
            8'd77: begin off = -24'sd2398; slope = -24'sd515; end
            8'd78: begin off = -24'sd2913; slope = -24'sd611; end
            8'd79: begin off = -24'sd3524; slope = -24'sd722; end
            8'd80: begin off = -24'sd4246; slope = -24'sd849; end
            8'd81: begin off = -24'sd5096; slope = -24'sd994; end
            8'd82: begin off = -24'sd6090; slope = -24'sd1159; end
            8'd83: begin off = -24'sd7249; slope = -24'sd1344; end
            8'd84: begin off = -24'sd8592; slope = -24'sd1551; end
            8'd85: begin off = -24'sd10143; slope = -24'sd1782; end
            8'd86: begin off = -24'sd11925; slope = -24'sd2037; end
            8'd87: begin off = -24'sd13962; slope = -24'sd2317; end
            8'd88: begin off = -24'sd16278; slope = -24'sd2622; end
            8'd89: begin off = -24'sd18900; slope = -24'sd2952; end
            8'd90: begin off = -24'sd21852; slope = -24'sd3306; end
            8'd91: begin off = -24'sd25158; slope = -24'sd3683; end
            8'd92: begin off = -24'sd28841; slope = -24'sd4081; end
            8'd93: begin off = -24'sd32922; slope = -24'sd4497; end
            8'd94: begin off = -24'sd37419; slope = -24'sd4926; end
            8'd95: begin off = -24'sd42346; slope = -24'sd5365; end
            8'd96: begin off = -24'sd47710; slope = -24'sd5807; end
            8'd97: begin off = -24'sd53517; slope = -24'sd6245; end
            8'd98: begin off = -24'sd59762; slope = -24'sd6671; end
            8'd99: begin off = -24'sd66433; slope = -24'sd7076; end
            8'd100: begin off = -24'sd73509; slope = -24'sd7451; end
            8'd101: begin off = -24'sd80960; slope = -24'sd7783; end
            8'd102: begin off = -24'sd88743; slope = -24'sd8062; end
            8'd103: begin off = -24'sd96805; slope = -24'sd8274; end
            8'd104: begin off = -24'sd105079; slope = -24'sd8405; end
            8'd105: begin off = -24'sd113484; slope = -24'sd8442; end
            8'd106: begin off = -24'sd121926; slope = -24'sd8372; end
            8'd107: begin off = -24'sd130298; slope = -24'sd8179; end
            8'd108: begin off = -24'sd138477; slope = -24'sd7851; end
            8'd109: begin off = -24'sd146328; slope = -24'sd7374; end
            8'd110: begin off = -24'sd153702; slope = -24'sd6735; end
            8'd111: begin off = -24'sd160437; slope = -24'sd5925; end
            8'd112: begin off = -24'sd166362; slope = -24'sd4933; end
            8'd113: begin off = -24'sd171295; slope = -24'sd3752; end
            8'd114: begin off = -24'sd175048; slope = -24'sd2377; end
            8'd115: begin off = -24'sd177424; slope = -24'sd803; end
            8'd116: begin off = -24'sd178227; slope = 24'sd970; end
            8'd117: begin off = -24'sd177257; slope = 24'sd2940; end
            8'd118: begin off = -24'sd174316; slope = 24'sd5103; end
            8'd119: begin off = -24'sd169213; slope = 24'sd7451; end
            8'd120: begin off = -24'sd161763; slope = 24'sd9973; end
            8'd121: begin off = -24'sd151789; slope = 24'sd12658; end
            8'd122: begin off = -24'sd139132; slope = 24'sd15488; end
            8'd123: begin off = -24'sd123644; slope = 24'sd18447; end
            8'd124: begin off = -24'sd105197; slope = 24'sd21514; end
            8'd125: begin off = -24'sd83683; slope = 24'sd24666; end
            8'd126: begin off = -24'sd59017; slope = 24'sd27882; end
            8'd127: begin off = -24'sd31135; slope = 24'sd31135; end
            8'd128: begin off = 24'sd0; slope = 24'sd34401; end
            8'd129: begin off = 24'sd34401; slope = 24'sd37654; end
            8'd130: begin off = 24'sd72055; slope = 24'sd40870; end
            8'd131: begin off = 24'sd112925; slope = 24'sd44022; end
            8'd132: begin off = 24'sd156947; slope = 24'sd47089; end
            8'd133: begin off = 24'sd204036; slope = 24'sd50048; end
            8'd134: begin off = 24'sd254084; slope = 24'sd52878; end
            8'd135: begin off = 24'sd306963; slope = 24'sd55563; end
            8'd136: begin off = 24'sd362525; slope = 24'sd58085; end
            8'd137: begin off = 24'sd420611; slope = 24'sd60433; end
            8'd138: begin off = 24'sd481044; slope = 24'sd62596; end
            8'd139: begin off = 24'sd543639; slope = 24'sd64566; end
            8'd140: begin off = 24'sd608205; slope = 24'sd66339; end
            8'd141: begin off = 24'sd674544; slope = 24'sd67913; end
            8'd142: begin off = 24'sd742456; slope = 24'sd69288; end
            8'd143: begin off = 24'sd811745; slope = 24'sd70469; end
            8'd144: begin off = 24'sd882214; slope = 24'sd71461; end
            8'd145: begin off = 24'sd953675; slope = 24'sd72271; end
            8'd146: begin off = 24'sd1025946; slope = 24'sd72910; end
            8'd147: begin off = 24'sd1098856; slope = 24'sd73387; end
            8'd148: begin off = 24'sd1172243; slope = 24'sd73715; end
            8'd149: begin off = 24'sd1245958; slope = 24'sd73908; end
            8'd150: begin off = 24'sd1319866; slope = 24'sd73978; end
            8'd151: begin off = 24'sd1393844; slope = 24'sd73941; end
            8'd152: begin off = 24'sd1467785; slope = 24'sd73810; end
            8'd153: begin off = 24'sd1541595; slope = 24'sd73598; end
            8'd154: begin off = 24'sd1615193; slope = 24'sd73319; end
            8'd155: begin off = 24'sd1688512; slope = 24'sd72987; end
            8'd156: begin off = 24'sd1761499; slope = 24'sd72612; end
            8'd157: begin off = 24'sd1834111; slope = 24'sd72207; end
            8'd158: begin off = 24'sd1906318; slope = 24'sd71781; end
            8'd159: begin off = 24'sd1978099; slope = 24'sd71343; end
            8'd160: begin off = 24'sd2049442; slope = 24'sd70901; end
            8'd161: begin off = 24'sd2120342; slope = 24'sd70462; end
            8'd162: begin off = 24'sd2190805; slope = 24'sd70033; end
            8'd163: begin off = 24'sd2260838; slope = 24'sd69617; end
            8'd164: begin off = 24'sd2330455; slope = 24'sd69219; end
            8'd165: begin off = 24'sd2399674; slope = 24'sd68842; end
            8'd166: begin off = 24'sd2468516; slope = 24'sd68488; end
            8'd167: begin off = 24'sd2537004; slope = 24'sd68158; end
            8'd168: begin off = 24'sd2605162; slope = 24'sd67853; end
            8'd169: begin off = 24'sd2673014; slope = 24'sd67573; end
            8'd170: begin off = 24'sd2740587; slope = 24'sd67318; end
            8'd171: begin off = 24'sd2807905; slope = 24'sd67087; end
            8'd172: begin off = 24'sd2874992; slope = 24'sd66880; end
            8'd173: begin off = 24'sd2941871; slope = 24'sd66695; end
            8'd174: begin off = 24'sd3008566; slope = 24'sd66530; end
            8'd175: begin off = 24'sd3075096; slope = 24'sd66385; end
            8'd176: begin off = 24'sd3141482; slope = 24'sd66258; end
            8'd177: begin off = 24'sd3207740; slope = 24'sd66147; end
            8'd178: begin off = 24'sd3273887; slope = 24'sd66051; end
            8'd179: begin off = 24'sd3339938; slope = 24'sd65968; end
            8'd180: begin off = 24'sd3405906; slope = 24'sd65897; end
            8'd181: begin off = 24'sd3471802; slope = 24'sd65836; end
            8'd182: begin off = 24'sd3537638; slope = 24'sd65784; end
            8'd183: begin off = 24'sd3603422; slope = 24'sd65740; end
            8'd184: begin off = 24'sd3669162; slope = 24'sd65704; end
            8'd185: begin off = 24'sd3734866; slope = 24'sd65673; end
            8'd186: begin off = 24'sd3800539; slope = 24'sd65647; end
            8'd187: begin off = 24'sd3866186; slope = 24'sd65626; end
            8'd188: begin off = 24'sd3931812; slope = 24'sd65609; end
            8'd189: begin off = 24'sd3997421; slope = 24'sd65594; end
            8'd190: begin off = 24'sd4063015; slope = 24'sd65583; end
            8'd191: begin off = 24'sd4128598; slope = 24'sd65573; end
            8'd192: begin off = 24'sd4194171; slope = 24'sd65565; end
            8'd193: begin off = 24'sd4259737; slope = 24'sd65559; end
            8'd194: begin off = 24'sd4325296; slope = 24'sd65554; end
            8'd195: begin off = 24'sd4390850; slope = 24'sd65550; end
            8'd196: begin off = 24'sd4456400; slope = 24'sd65547; end
            8'd197: begin off = 24'sd4521948; slope = 24'sd65545; end
            8'd198: begin off = 24'sd4587492; slope = 24'sd65543; end
            8'd199: begin off = 24'sd4653035; slope = 24'sd65541; end
            8'd200: begin off = 24'sd4718576; slope = 24'sd65540; end
            8'd201: begin off = 24'sd4784116; slope = 24'sd65539; end
            8'd202: begin off = 24'sd4849655; slope = 24'sd65538; end
            8'd203: begin off = 24'sd4915193; slope = 24'sd65538; end
            8'd204: begin off = 24'sd4980731; slope = 24'sd65537; end
            8'd205: begin off = 24'sd5046268; slope = 24'sd65537; end
            8'd206: begin off = 24'sd5111805; slope = 24'sd65537; end
            8'd207: begin off = 24'sd5177342; slope = 24'sd65537; end
            8'd208: begin off = 24'sd5242878; slope = 24'sd65536; end
            8'd209: begin off = 24'sd5308415; slope = 24'sd65536; end
            8'd210: begin off = 24'sd5373951; slope = 24'sd65536; end
            8'd211: begin off = 24'sd5439487; slope = 24'sd65536; end
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
            8'd77: begin off = -24'sd1; slope = 24'sd0; end
            8'd78: begin off = -24'sd1; slope = 24'sd0; end
            8'd79: begin off = -24'sd1; slope = 24'sd0; end
            8'd80: begin off = -24'sd1; slope = 24'sd0; end
            8'd81: begin off = -24'sd1; slope = 24'sd0; end
            8'd82: begin off = -24'sd1; slope = 24'sd0; end
            8'd83: begin off = -24'sd2; slope = 24'sd0; end
            8'd84: begin off = -24'sd2; slope = 24'sd0; end
            8'd85: begin off = -24'sd2; slope = 24'sd0; end
            8'd86: begin off = -24'sd3; slope = 24'sd0; end
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
            8'd169: begin off = 24'sd653; slope = 24'sd16; end
            8'd170: begin off = 24'sd669; slope = 24'sd16; end
            8'd171: begin off = 24'sd686; slope = 24'sd16; end
            8'd172: begin off = 24'sd702; slope = 24'sd16; end
            8'd173: begin off = 24'sd718; slope = 24'sd16; end
            8'd174: begin off = 24'sd735; slope = 24'sd16; end
            8'd175: begin off = 24'sd751; slope = 24'sd16; end
            8'd176: begin off = 24'sd767; slope = 24'sd16; end
            8'd177: begin off = 24'sd783; slope = 24'sd16; end
            8'd178: begin off = 24'sd799; slope = 24'sd16; end
            8'd179: begin off = 24'sd815; slope = 24'sd16; end
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
