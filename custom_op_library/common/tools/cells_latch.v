// Techmap for synth.py: yosys's internal latches onto the Nangate45 latch cells
// (dfflibmap maps flip-flops only). Active-high enable -> DLH_X1, active-low -> DLL_X1.
module \$_DLATCH_P_ (input E, input D, output Q);
    DLH_X1 _TECHMAP_REPLACE_ (.D(D), .G(E), .Q(Q));
endmodule

module \$_DLATCH_N_ (input E, input D, output Q);
    DLL_X1 _TECHMAP_REPLACE_ (.D(D), .GN(E), .Q(Q));
endmodule
