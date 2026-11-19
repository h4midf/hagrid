* Testbench: full 8-bit SAR ADC (Stage B) = input buffer + bootstrapped switch + core.
* sar_full_tb.sp (7 bits) with one more SAR bit: the same unit capacitor, DAC switches,
* comparator and front end; one more shift-register stage, bit register and binary
* capacitor (128 cu), so 9 cycles per conversion (1 sample + 8 decisions).
* The bitline voltage steps to a new value at the start of every conversion
* (bitline settles, then the ADC samples).
* Conversion k occupies [k*tconv, (k+1)*tconv]; conversion 0 flushes the logic.
* Runner substitutes IBUF (buffer bias), the input, and __DESIGN__ (supply, Vref,
* clock period, conversion time from design.json via tb.py).
.include ../models/32nm_LP.pm
.include ../models/32nm_HP_lvt.pm
.include sar_core.lib
.include sar_frontend.lib
.param cu=2f
.param ibuf=1u
__DESIGN__

VDD  vdd  0 {vdd_v}
VDDA vdda 0 {vdd_v}
VREF vref 0 {vref_v}
VCM  vcm  0 0.3
VBL  bl   0 PWL(__PWL__)

VCLK clk 0 PULSE(0 {vdd_v} 0 0.1n 0.1n {tclk/2-0.1n} {tclk})
VSAMPT sampt 0 PULSE(0 {vdd_v} 2n 0.1n 0.1n {tclk-1.1n} {tconv})
VSAMPB samp  0 PULSE(0 {vdd_v} 2n 0.1n 0.1n {tclk-0.1n} {tconv})
Xis1 sampt samptb vdd inv
Xis2 samp  sampb  vdd inv
Xic  clk   clkc   vdd inv

* front end: buffer (own supply VDDA) -> bootstrapped switch -> sampling bus
* sar_full_tb.sp's front end with the buffer re-sized for the 2x DAC load and the 8-bit
* LSB (run_frontend_resize.py, results/adc.json frontend_resize of the 128 config: no
* point within 0.5 LSB, this one has the smallest worst sample error, 0.53 LSB);
* the buffer bias comes from design.json adc.buffer_bias_uA
Xbuf bl vbuf vdda buffer_ab IB={ibuf} CC=75f N2=8 N6=12 CLS=200f
Xbs vbuf vin samp sampb vdd bootsw CB=25f

Xr1 samp clk t7 vdd dff
Xr2 t7   clk t6 vdd dff
Xr3 t6   clk t5 vdd dff
Xr4 t5   clk t4 vdd dff
Xr5 t4   clk t3 vdd dff
Xr6 t3   clk t2 vdd dff
Xr7 t2   clk t1 vdd dff
Xr8 t1   clk t0 vdd dff

Xb7 t7 cmp samp clk b7 d7 vdd sarbit
Xb6 t6 cmp samp clk b6 d6 vdd sarbit
Xb5 t5 cmp samp clk b5 d5 vdd sarbit
Xb4 t4 cmp samp clk b4 d4 vdd sarbit
Xb3 t3 cmp samp clk b3 d3 vdd sarbit
Xb2 t2 cmp samp clk b2 d2 vdd sarbit
Xb1 t1 cmp samp clk b1 d1 vdd sarbit
Xb0 t0 cmp samp clk b0 d0 vdd sarbit

C7 top bot7 {128*cu}
C6 top bot6 {64*cu}
C5 top bot5 {32*cu}
C4 top bot4 {16*cu}
C3 top bot3 {8*cu}
C2 top bot2 {4*cu}
C1 top bot1 {2*cu}
C0 top bot0 {1*cu}
CD top botd {1*cu}
Xs7 bot7 vin vref samp sampb d7 vdd dacsw
Xs6 bot6 vin vref samp sampb d6 vdd dacsw
Xs5 bot5 vin vref samp sampb d5 vdd dacsw
Xs4 bot4 vin vref samp sampb d4 vdd dacsw
Xs3 bot3 vin vref samp sampb d3 vdd dacsw
Xs2 bot2 vin vref samp sampb d2 vdd dacsw
Xs1 bot1 vin vref samp sampb d1 vdd dacsw
Xs0 bot0 vin vref samp sampb d0 vdd dacsw
Xtd vin botd samp sampb vdd tglvt W=192n
Mdg botd sampb 0 0 nmos_lvt W=192n L=32n
Xtt vcm top sampt samptb vdd tglvt W=192n
Xcmp top vcm clkc cmp vdd comparator

.tran 20p __TEND__
.control
run
let code = 128*(v(b7)>0.5)+64*(v(b6)>0.5)+32*(v(b5)>0.5)+16*(v(b4)>0.5)+8*(v(b3)>0.5)+4*(v(b2)>0.5)+2*(v(b1)>0.5)+(v(b0)>0.5)
__MEAS__
.endc
.end
