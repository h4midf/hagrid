# HyFlexPIM `default`: reconstructed vs Song et al. 2025 (Table 2)

Generated 2026-11-17T00:51:42 by report.py from results/*.json.

| Quantity | Ours | HyFlexPIM | Status |
|---|---|---|---|
| Stillmaker area factor, 32 nm HP -> 65 nm bulk | 3.255 | 3.3 (Table 4, rounded) | match |
| Stillmaker energy factor, 1.0 V both nodes | 2.326 | - | ours only (fits Table 2) |
| analog module rram_array, mm^2 | 0.04514 | 0.048 | gap: x1.064 in Table 2, cause not found |
| analog module rram_array, mW | 60.78 | 60.78 | match |
| analog module ir, mm^2 | 0.0006511 | 0.00065 | match |
| analog module ir, mW | 0.1326 | 0.13 | match |
| analog module or, mm^2 | 0.001302 | 0.00129 | within 0.9% |
| analog module or, mW | 0.5303 | 0.53 | match |
| analog module wl_drv, mm^2 | 0.01771 | 0.02 | match |
| analog module wl_drv, mW | 297.7 | 297.71 | match |
| analog module adc, mm^2 | 0.2839 | 0.3 | match |
| analog module adc, mW | 511 | 512 | within 0.2% |
| analog module s_and_a, mm^2 | 0.1002 | 0.1 | match |
| analog module s_and_a, mW | 56.27 | 59.54 | within 5.5% |
| analog module s_and_h, mm^2 | 6.511e-05 | 6e-05 | within 8.5% |
| analog module s_and_h, mW | 0.01163 | 1.2e-05 | Table 2 printed in W: ours 11.6e-6 W |
| analog module sum, mm^2 | 0.4491 | 0.47 | x0.955 |
| analog module sum, mW | 926.4 | 930.69 | x0.995 |
| digital module rram_array, mm^2 | 2.865 | 2.86 | match |
| digital module rram_array, mW | 3890 | 3890.02 | match |
| digital module ir, mm^2 | 0.003093 | 0.0031 | match |
| digital module ir, mW | 0.7592 | 0.76 | match |
| digital module or, mm^2 | 0.003276 | 0.0032 | within 2.4% |
| digital module or, mW | 1.667 | 1.65 | within 1.1% |
| digital module wl_drv, mm^2 | 0.1417 | 0.14 | match |
| digital module wl_drv, mW | 2382 | 2381.64 | match |
| digital module s_and_a, mm^2 | 0.2005 | 0.21 | within 4.5% |
| digital module s_and_a, mW | 105.8 | 119.08 | gap: ours x0.89 |
| digital module s_and_h, mm^2 | 0.0001302 | 0.00013 | match |
| digital module s_and_h, mW | 0.02326 | 2.3e-05 | Table 2 printed in W: ours 23.3e-6 W |
| digital module sfu, mm^2 | 5.332 | 4.79 | gap: ours x1.11 |
| digital module sfu, mW | 127.2 | 138.89 | within 8.4% |
| digital module sum, mm^2 | 8.546 | 8.01 | x1.067 |
| digital module sum, mW | 6507 | 6532.05 | x0.996 |
| digital module sum without SFU, mm^2 | 3.213 | 3.22 | x0.998 |
| digital module sum without SFU, mW | 6380 | 6393.16 | x0.998 |
| SFU dynamic power, BERT-Large N=128 workload, idle units clock-gated, mW | 34.08 | - | ours only; busy 2.8% of the layer time |
| SFU leakage power, mW | 93.13 | - | ours only |
| SFU power with every unit busy every cycle (alpha 0.2), mW | 1240 | - | ours only; not what Table 2 reports |
| SFU pipeline stage at 65 nm (logic + flop), ps | 934 | 1000 (1 GHz, Sec. 5.4) | meets 1 GHz |
| ISAAC S+A (bare 16-bit shift-and-add), 32 nm, um^2 | 60.14 | 60 (ISAAC Table I) | x1.00: our flow reproduces the source |
| ISAAC S+A, 32 nm, 1.2 GHz, mW | 0.04259 | 0.05 (ISAAC Table I) | x0.85 (alpha 0.2) |
| Working S&A per array (column weighting, partial, accumulate, 1.28 GHz), um^2 | 1468 | 195.3 (Table 2 / 512) | ours x7.5: Table 2's row is one bare adder |
| Working S&A per array, mW | 0.545 | 0.1163 (Table 2 / 512) | ours x4.7 |
| Working S&A stage at 65 nm, ps | 697 | 781 (1.28 GS/s ADC) | meets the ADC rate |
| ISAAC IR (1024 B SRAM, 32 nm, CACTI 6.5), mm^2 | 0.002044 | 0.0021 (ISAAC Table I) | x0.97 |
| ISAAC OR (256 B SRAM, 32 nm, CACTI 6.5), mm^2 | 0.0006584 | 0.00077 (ISAAC Table I) | x0.86 |
| ADC scan of one array (128 columns / 1.28 GS/s) | 100.0 ns | 100 ns | match; the 100 ns crossbar read is ISAAC's assumption, not simulated |
