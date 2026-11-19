# Hagrid

A compiler that maps PyTorch models onto hybrid compute-in-memory hardware (1F1C,
HyFlexPIM, RSE), with the hardware models and the scripts behind the paper's results.

## Results

Every result is already in the repository:

- `reproduce/results/*.csv`: each paper value next to the value regenerated from the code
- `targets/<target>/**/results/`: the hardware models' simulation and synthesis results
- `run_scripts/results/`: the mapping sweeps (greedy, list scheduling, CP-SAT)
- `dump/gpu_baseline/`: the GPU measurements

## Reproducing

Setup: `./setup.sh` installs the Python packages into `.venv` and builds Yosys 0.68 and
ngspice-47 into `.tools` (no root). It needs make, a C/C++ compiler (g++ and clang), m4,
curl and tar.

Each script below skips what already exists, compares the result with the paper and
writes a CSV to `reproduce/results/`. `--force` deletes its results and runs it from
scratch. Run from the repo root.

**Hardware validation table** (about 25 min, mostly the 1F1C SPICE runs):

```bash
python3 reproduce/hardware_validation_1f1c.py
python3 reproduce/hardware_validation_hyflexpim.py
python3 reproduce/hardware_validation_rse.py
```

**Mapping latency figure**, per target (`fefet1c`, `hyflexpim`, `rse`):

```bash
python3 reproduce/mapping_<target>_greedy.py --workers 8
python3 reproduce/mapping_<target>_listsched.py           # after greedy
python3 reproduce/mapping_<target>_cpsat.py --cpsat-workers 24   # 1 h per case, up to 24 h per target
```

GPU lines (on a machine with that NVIDIA GPU and `run_scripts/gpu_baseline/requirements.txt`):

```bash
python reproduce/gpu_baseline.py --device rtx3090 --label "NVIDIA GeForce RTX 3090" --cpus 8-11 --dry-model resnet50
```

Then draw the figure from those results:

```bash
python3 reproduce/mapping_latency.py
```
