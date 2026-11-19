# custom_op_library

Digital CMOS logic for the custom-chip operations. One folder per operation: hand-written Verilog in `rtl/`, `characterize*.py` scripts that synthesize it, and the numbers in `results/`. Shared tools are in `common/`.

Only CMOS logic lives here. LUT table values are memory and are not part of this library. Matmul runs on the CIM array and is not here.

Units: area in um^2 (NanGate45 typical, 45 nm; 28 nm = 0.35 x), delay in ps, leakage in uW, dynamic power as an estimate in mW.
