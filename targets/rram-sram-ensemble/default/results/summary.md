# RRAM/SRAM ensemble `default`: reconstructed vs Krishnan et al. 2022

Generated 2026-11-12T00:04:16 by report.py from results/*.json. 'match' = within 15% of the paper.

| Quantity | Ours | Paper | Status |
|---|---|---|---|
| RRAM MAC per input bit (8-to-1 column mux at 100 MHz) | 8 cycles, 80 ns | 8 cycles (Sec. III-D-1) | match (paper's own value) |
| Header + ADC share of RRAM macro power, 1 rows on (all column current as the header's), % | 87.5 | 90 (Sec. V-F) | match (3% off) |
| Header + ADC share of RRAM macro power, 8 rows on (all column current as the header's), % | 88.3 | 90 (Sec. V-F) | match (2% off) |
| Header + ADC share of RRAM macro power, 16 rows on (all column current as the header's), % | 86.3 | 90 (Sec. V-F) | match (4% off) |
| Header + ADC share of RRAM macro power, 32 rows on (all column current as the header's), % | 84.6 | 90 (Sec. V-F) | match (6% off) |
| Header + ADC share of RRAM macro power, 64 rows on (all column current as the header's), % | 99.0 | 90 (Sec. V-F) | match (10% off) |
| SRAM over RRAM area, ResNet-20 CIFAR-10, % | 2.62 (Fig. 6 layout: 2.05) | 3.7 (Fig. 12a) | gap: ours x0.71 |
| SRAM over RRAM area, VGG-16 CIFAR-10, % | 2.58 (Fig. 6 layout: 2.02) | 2.3 (Fig. 12a) | match (12% off) |
| SRAM over RRAM area, MobileNet-v2 ImageNet, % | 15.72 (Fig. 6 layout: 12.31) | 20 (Fig. 12a) | gap: ours x0.79 |
| SRAM over RRAM power, ResNet-20 CIFAR-10, 16 rows on, % | 0.50 (8/16/32 rows: 0.69/0.50/0.43) | 0.5 (Fig. 12b) | match (0% off) |
| SRAM over RRAM power, VGG-16 CIFAR-10, 16 rows on, % | 0.49 (8/16/32 rows: 0.68/0.49/0.42) | 0.3 (Fig. 12b) | gap: ours x1.64 |
| SRAM over RRAM power, MobileNet-v2 ImageNet, 16 rows on, % | 3.00 (8/16/32 rows: 4.14/3.00/2.58) | 2.6 (Fig. 12b) | gap: ours x1.15 |
| Single-macro area ratio (SRAM / RRAM) | 0.1098 (Fig. 6 layout: 0.0860) | 0.1549 / 0.0981 / 0.1398 (implied per network) | best any fixed macro: 0.1201 at best, 22% worst-case error |
| Single-macro power ratio (SRAM / RRAM), 16 rows on | 0.0210 | 0.0209 / 0.0128 / 0.0182 (implied per network) | best any fixed macro: 0.0159 at best, 24% worst-case error |
| SRAM macro (logic + array), um^2 / mW | 7599 / 0.1640 | - | ours only |
| RRAM macro, um^2 / mW at 16 rows on | 69177 / 7.816 | - | ours only |
| Header + ADC share at a 0.1 V read supply, 16 rows on, % | 15.6 | - | ours only; no pMOS header reaches it |
| Header + ADC share at a 0.2 V read supply, 16 rows on, % | 25.2 | - | ours only; no pMOS header reaches it |
| Header + ADC share at a 0.3 V read supply, 16 rows on, % | 29.9 | - | ours only; header 75300 nm |
| Header + ADC share at a 0.45 V read supply, 16 rows on, % | 46.7 | - | ours only; header 8992 nm |
| Header + ADC share at a 0.6 V read supply, 16 rows on, % | 60.9 | - | ours only; header 3592 nm |
| Header + ADC share at a 0.9 V read supply, 16 rows on, % | 77.9 | - | ours only; header 1854 nm |
| Header + ADC share at a 1.2 V read supply, 16 rows on, % | 86.1 | - | ours only; header 1415 nm |
| Accuracy, ResNet-20 CIFAR-10, % | not run | 90.73 (Table III) | not run |
| Accuracy, VGG-16 CIFAR-10, % | not run | 92.75 (Table III) | not run |
| Accuracy, ResNet-18 ImageNet, % | not run | 69.21 (Table III) | not run |
| Accuracy, MobileNet-v2 ImageNet, % | not run | 61.6 (Table III) | not run |
