* The RRAM macro's 3-bit flash ADC (Krishnan et al., TCAD 2022, Sec. III-D-1): 7 StrongARM
* comparators against a resistor ladder, latched, then a thermometer-to-binary encoder.
* The paper gives only "a flash ADC with a 3-bit resolution"; the comparators, ladder and
* encoder are our own design. References below mid-rail use PMOS-input comparators,
* the others NMOS-input, since the bitline voltage spans most of the supply.
* The double-underscore placeholders are filled in by characterize_readout.py.
.include __MODELS__
.include __CELLS__
.param tck=__TCK__ vdd=__VDD__ rseg=__RSEG__ ctap=__CTAP__ vrefv=__VREF__

VDD  vdd 0 {vdd}
VREF vref 0 {vrefv}
VCK  clk 0 PULSE(0 {vdd} {tck/2} 50p 50p {tck/2-50p} {tck})
VIN  vin 0 PWL(__PWL__)
Xcb  clk clkb vdd inv s=4

* reference ladder: tap k at k / 8 of the read voltage
R0 0  r1 {rseg}
R1 r1 r2 {rseg}
R2 r2 r3 {rseg}
R3 r3 r4 {rseg}
R4 r4 r5 {rseg}
R5 r5 r6 {rseg}
R6 r6 r7 {rseg}
R7 r7 vref {rseg}
* tap decoupling: without it, comparator clock kickback moves a tap by over 200 mV
CD1 r1 0 {ctap}
CD2 r2 0 {ctap}
CD3 r3 0 {ctap}
CD4 r4 0 {ctap}
CD5 r5 0 {ctap}
CD6 r6 0 {ctap}
CD7 r7 0 {ctap}

XC1 vin r1 clkb t1 t1b vdd cmp_p
XC2 vin r2 clkb t2 t2b vdd cmp_p
XC3 vin r3 clkb t3 t3b vdd cmp_p
XC4 vin r4 clk t4 t4b vdd cmp_n
XC5 vin r5 clk t5 t5b vdd cmp_n
XC6 vin r6 clk t6 t6b vdd cmp_n
XC7 vin r7 clk t7 t7b vdd cmp_n

* encoder: b2 = t4; b1 = t6 + t2 !t4; b0 = t7 + t5 !t6 + t3 !t4 + t1 !t2
XB2 t4b b2 vdd inv
XE1 t2 t4b n1 vdd nand2
XB1 t6b n1 b1 vdd nand2
XE2 t5 t6b n2 vdd nand2
XE3 t3 t4b n3 vdd nand2
XE4 t1 t2b n4 vdd nand2
XB0 t7b n2 n3 n4 b0 vdd nand4

.tran 10p __TEND__
.control
run
__MEAS__
.endc
.end
