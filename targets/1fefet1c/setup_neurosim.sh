#!/usr/bin/env bash
# Set up NeuroSim (DNN_NeuroSim_V1.4) on this machine: Python venv + C++ build.
# Works on Linux (GCC, CUDA GPU if present) and macOS (Apple clang + Homebrew libomp).
#
# Usage (from anywhere):  bash targets/1fefet1c/setup_neurosim.sh
# Then:                   python3 targets/1fefet1c/run_neurosim.py default
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
NS="$HERE/NeuroSim"
PY="${PYTHON:-python3}"   # override with PYTHON=python3.11 if needed

echo "== Python venv: $NS/.venv (using $($PY --version))"
if [ ! -x "$NS/.venv/bin/python" ]; then
    "$PY" -m venv "$NS/.venv"
fi
"$NS/.venv/bin/pip" install -q --upgrade pip
# On Linux x86_64 the default PyPI torch wheel includes CUDA.
"$NS/.venv/bin/pip" install -q torch torchvision numpy tqdm lmdb
"$NS/.venv/bin/python" - <<'EOF'
import torch, torchvision
print("torch", torch.__version__, "torchvision", torchvision.__version__)
print("CUDA available:", torch.cuda.is_available(),
      "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-> NeuroSim will run on CPU")
EOF

echo "== Building NeuroSim C++ ($NS/Inference_pytorch/NeuroSIM)"
cd "$NS/Inference_pytorch/NeuroSIM"
make clean >/dev/null
if [ "$(uname)" = "Darwin" ]; then
    O="$(brew --prefix libomp)"
    # Apple clang: OpenMP via libomp; newer clang errors on a chained comparison
    # in CurrentSenseAmp.cpp (GCC only warns), so keep it a warning.
    make -j"$(sysctl -n hw.ncpu)" CXXFLAGS="-Xpreprocessor -fopenmp -I$O/include -L$O/lib -lomp -O3 -std=c++0x -w -Wno-error=parentheses" >/dev/null
else
    make -j"$(nproc)" >/dev/null
fi
ls -la main
echo "== Done. Run: python3 $HERE/run_neurosim.py default"
