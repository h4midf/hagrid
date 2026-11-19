* One channel of the HyFlexPIM ADC: 7-bit asynchronous SAR with alternate comparators
* (the architecture of Kull et al., JSSC 2013, HyFlexPIM's ADC source [29]), PTM 65 nm
* bulk, VDD = Vref = 1.0 V. HyFlexPIM's 1.28 GS/s ADC is two of these, interleaved
* (characterize_adc.py).
* Single-ended input (the column sample-and-hold through the multiplexer, HyFlexPIM
* Fig. 8) on the P side; the N side samples a fixed mid-scale reference.
* Decisions k = 6 (MSB) .. 0; comparator A takes k = 6, 4, 2, 0, comparator B k = 5, 3, 1.
* Set-and-up CDAC, 1 fF units (Kull's unit; 64 fF per side for 7 bits): decision k
* (k >= 1) raises the lower side's capacitor of 2^(k-1) units from 0 to Vref.
* Asynchronous loop: decision k done (valid of its comparator while p_k) sets flag f_k;
* the pointer p_k = f_(k+1) and not f_k selects the current decision; a comparator's
* clock is the OR of its pending pointers. Latches reset while sampling.
* The model and cell paths, input waveforms, end time and measurements (the
* double-underscore placeholders below) are filled in by characterize_adc.py.
.include __MODELS__
.include __CELLS__
.param tck=1562.5p tsamp=200p

VDD  vdd  0 1.0
VREF vref 0 1.0
VCK  samp 0 PULSE(0 1.0 0 10p 10p {tsamp} {tck})
VINP vinp 0 PWL(__PWL_P__)
VINN vinn 0 PWL(__PWL_N__)

* T&H switch enable (sampb) and the convert phase (conv = not samp) on separate chains
Xsb samp sampb vdd inv s=16
Xc1 samp c1 vdd inv s=4
Xcv c1 convn vdd inv s=8
Xcv2 convn conv vdd inv s=16
* track and hold
XTP vinp topp samp sampb vdd tg wn=8.1u wp=16.3u
XTN vinn topn samp sampb vdd tg wn=8.1u wp=16.3u

* ---- decision pointer and comparator clocks
Xp6 convn f6 p6 vdd nor2
Xp5 f6b f5 p5 vdd nor2
Xp4 f5b f4 p4 vdd nor2
Xp3 f4b f3 p3 vdd nor2
Xp2 f3b f2 p2 vdd nor2
Xp1 f2b f1 p1 vdd nor2
Xp0 f1b f0 p0 vdd nor2
XpA1 p6 p4 nA1 vdd nor2 s=2
XpA2 p2 p0 nA2 vdd nor2 s=2
XpA3 nA1 nA2 clkA vdd nand2 s=6
XpB1 p5 p3 nB1 vdd nor2 s=2
XpB2 p1 0 nB2 vdd nor2 s=2
XpB3 nB1 nB2 clkB vdd nand2 s=6

* ---- comparators
XCA topp topn clkA dpA dnA validA vdd strongarm w=2
XCB topp topn clkB dpB dnB validB vdd strongarm w=2

* ---- per decision: done flag f_k, CDAC controls cN_k (raise N side: p > n) and cP_k;
* bit k = cN_k. NAND latches, set active low, reset (rn = conv) while sampling.
XsF6 validA p6 sF6 vdd nand2
XsN6 validA p6 dpA sN6 vdd nand3
XsP6 validA p6 dnA sP6 vdd nand3
XlF6 sF6 conv f6 f6b vdd srln
XlN6 sN6 conv cN6 cN6b vdd srln
XlP6 sP6 conv cP6 cP6b vdd srln
XsF5 validB p5 sF5 vdd nand2
XsN5 validB p5 dpB sN5 vdd nand3
XsP5 validB p5 dnB sP5 vdd nand3
XlF5 sF5 conv f5 f5b vdd srln
XlN5 sN5 conv cN5 cN5b vdd srln
XlP5 sP5 conv cP5 cP5b vdd srln
XsF4 validA p4 sF4 vdd nand2
XsN4 validA p4 dpA sN4 vdd nand3
XsP4 validA p4 dnA sP4 vdd nand3
XlF4 sF4 conv f4 f4b vdd srln
XlN4 sN4 conv cN4 cN4b vdd srln
XlP4 sP4 conv cP4 cP4b vdd srln
XsF3 validB p3 sF3 vdd nand2
XsN3 validB p3 dpB sN3 vdd nand3
XsP3 validB p3 dnB sP3 vdd nand3
XlF3 sF3 conv f3 f3b vdd srln
XlN3 sN3 conv cN3 cN3b vdd srln
XlP3 sP3 conv cP3 cP3b vdd srln
XsF2 validA p2 sF2 vdd nand2
XsN2 validA p2 dpA sN2 vdd nand3
XsP2 validA p2 dnA sP2 vdd nand3
XlF2 sF2 conv f2 f2b vdd srln
XlN2 sN2 conv cN2 cN2b vdd srln
XlP2 sP2 conv cP2 cP2b vdd srln
XsF1 validB p1 sF1 vdd nand2
XsN1 validB p1 dpB sN1 vdd nand3
XsP1 validB p1 dnB sP1 vdd nand3
XlF1 sF1 conv f1 f1b vdd srln
XlN1 sN1 conv cN1 cN1b vdd srln
XlP1 sP1 conv cP1 cP1b vdd srln
XsF0 validA p0 sF0 vdd nand2
XsN0 validA p0 dpA sN0 vdd nand3
XlF0 sF0 conv f0 f0b vdd srln
XlN0 sN0 conv cN0 cN0b vdd srln

* ---- CDAC, 1 fF units: decision k raises a capacitor of 2^(k-1) units; drivers sized
* with their capacitor
XDP6 cP6 botP6 vref vdd drv s=16
XDP5 cP5 botP5 vref vdd drv s=8
XDP4 cP4 botP4 vref vdd drv s=4
XDP3 cP3 botP3 vref vdd drv s=2
XDP2 cP2 botP2 vref vdd drv s=1
XDP1 cP1 botP1 vref vdd drv s=1
XDN6 cN6 botN6 vref vdd drv s=16
XDN5 cN5 botN5 vref vdd drv s=8
XDN4 cN4 botN4 vref vdd drv s=4
XDN3 cN3 botN3 vref vdd drv s=2
XDN2 cN2 botN2 vref vdd drv s=1
XDN1 cN1 botN1 vref vdd drv s=1
CP6 topp botP6 32f
CP5 topp botP5 16f
CP4 topp botP4 8f
CP3 topp botP3 4f
CP2 topp botP2 2f
CP1 topp botP1 1f
CPD topp 0 1f
CN6 topn botN6 32f
CN5 topn botN5 16f
CN4 topn botN4 8f
CN3 topn botN3 4f
CN2 topn botN2 2f
CN1 topn botN1 1f
CND topn 0 1f

.tran 1p __TEND__
.option method=gear
.control
run
__MEAS__
.endc
.end
