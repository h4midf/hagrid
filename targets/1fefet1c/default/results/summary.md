# 1F1C `default`: simulated vs Laleni et al. 2024

Generated 2026-11-07T21:31:57 by report.py from results/*.json.

| Quantity | Ours | Laleni | Status |
|---|---|---|---|
| Fig. 6 FeFET charging plateau (Vth 0.3 V, measured) | 0.705 V | ~0.70 V | match |
| Cpar, 128-row bitline | 75.0 fF | 75 fF | set: cell height fitted |
| kappa (charge-sharing ratio) | 0.729 | 0.69 | match |
| Full-scale bitline voltage | 0.482 V | ~0.5 V (Vrefp) | match |
| Bitline settling (1 mV) | 8.86 ns | - | ours only |
| Bitline charging power per cell (CMoM*V^2*f_conv) | 0.77 nW | 0.86 nW | match (accounting inferred) |
| Column charging energy per MAC (128 cells, VDD/2) | 117.6 fJ | - | ours only |
| Fig. 10(b) max VBL (4 cells, no wire) | 0.742 V | 0.90 V (eq. 3, calculated) | gap explained: 50 ns charging + cell parasitics |
| ADC energy per conversion | 2.165 pJ | 2.196 pJ (FoM*2^ENOB) | match |
| ADC ENOB (no noise/mismatch) | 6.12 bits | 6.4 bits | close; 64-sample record limits it |
| ADC sampling noise (buffer + kT/C) | 0.564 mV rms | - | ours only |
| ADC ENOB, Monte Carlo (mismatch + noise) | 6.00 +/- 0.24 bits (n=30) | 6.4 bits | ours only (Laleni's conditions unknown) |
| Efficiency | 112.2 TOPS/W | 111.6 TOPS/W | match (formula inferred) |
| CIFAR-10 VGG-8 accuracy, vari = 0.0 | 91.80 % | 89 % | NeuroSim |
| CIFAR-10 VGG-8 accuracy, vari = 0.06 | 91.46 % | 89 % | NeuroSim |
