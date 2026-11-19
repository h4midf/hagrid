* Testbench: 7-bit SAR ADC core with an ideal input (Stage A).
* 8 clock cycles per conversion: cycle 0 samples, cycles 1-7 decide B6..B0.
* VIN is set by the runner via .param vin_dc. Supply, Vref, clock period and
* conversion time come from design.json via tb.py (the __DESIGN__ line).
.include ../models/32nm_LP.pm
* low-Vt devices for analog switches (PTM 32nm HP, renamed nmos_lvt/pmos_lvt)
.include ../models/32nm_HP_lvt.pm
.include sar_core.lib
.param vin_dc=0.25
.param cu=2f
__DESIGN__

VDD vdd 0 {vdd_v}
VREF vref 0 {vref_v}
* VCM low so the top-plate NMOS switch turns on fully (LP Vth ~0.56 V)
VCM vcm 0 0.3
VIN vin 0 {vin_dc}

* clock: rising at n*tclk, high for the first half (DAC settles), low for the
* second half (comparator evaluates)
VCLK clk 0 PULSE(0 {vdd_v} 0 0.1n 0.1n {tclk/2-0.1n} {tclk})
* sampling: top plate opens at tclk+1 ns, bottom plates leave vin at tclk+2 ns
VSAMPT sampt 0 PULSE(0 {vdd_v} 2n 0.1n 0.1n {tclk-1.1n} {tconv})
VSAMPB samp  0 PULSE(0 {vdd_v} 2n 0.1n 0.1n {tclk-0.1n} {tconv})
Xis1 sampt samptb vdd inv
Xis2 samp  sampb  vdd inv
Xic  clk   clkc   vdd inv

* shift register: one-hot trial signals t6 (cycle 1) ... t0 (cycle 7)
Xr1 samp clk t6 vdd dff
Xr2 t6   clk t5 vdd dff
Xr3 t5   clk t4 vdd dff
Xr4 t4   clk t3 vdd dff
Xr5 t3   clk t2 vdd dff
Xr6 t2   clk t1 vdd dff
Xr7 t1   clk t0 vdd dff

* SAR bit registers
Xb6 t6 cmp samp clk b6 d6 vdd sarbit
Xb5 t5 cmp samp clk b5 d5 vdd sarbit
Xb4 t4 cmp samp clk b4 d4 vdd sarbit
Xb3 t3 cmp samp clk b3 d3 vdd sarbit
Xb2 t2 cmp samp clk b2 d2 vdd sarbit
Xb1 t1 cmp samp clk b1 d1 vdd sarbit
Xb0 t0 cmp samp clk b0 d0 vdd sarbit

* capacitor DAC (binary, unit cu) + dummy
C6 top bot6 {64*cu}
C5 top bot5 {32*cu}
C4 top bot4 {16*cu}
C3 top bot3 {8*cu}
C2 top bot2 {4*cu}
C1 top bot1 {2*cu}
C0 top bot0 {1*cu}
CD top botd {1*cu}
Xs6 bot6 vin vref samp sampb d6 vdd dacsw
Xs5 bot5 vin vref samp sampb d5 vdd dacsw
Xs4 bot4 vin vref samp sampb d4 vdd dacsw
Xs3 bot3 vin vref samp sampb d3 vdd dacsw
Xs2 bot2 vin vref samp sampb d2 vdd dacsw
Xs1 bot1 vin vref samp sampb d1 vdd dacsw
Xs0 bot0 vin vref samp sampb d0 vdd dacsw
Xtd vin botd samp sampb vdd tglvt W=192n
Mdg botd sampb 0 0 nmos_lvt W=192n L=32n

* top plate held at VCM while sampling
Xtt vcm top sampt samptb vdd tglvt W=192n

* comparator: cmp = 1 when top < VCM  (i.e. vin > DAC -> keep the bit)
Xcmp top vcm clkc cmp vdd comparator

.tran 20p __TEND__
.control
run
* conversions end at k*tconv; bits are valid until the next reset (runner adds the measures)
let code1 = 64*(v(b6)>0.5)+32*(v(b5)>0.5)+16*(v(b4)>0.5)+8*(v(b3)>0.5)+4*(v(b2)>0.5)+2*(v(b1)>0.5)+(v(b0)>0.5)
__MEAS__
.endc
.end
