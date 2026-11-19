#!/bin/bash
# Unattended GPU baseline run, on the GPU machine, from anywhere:
#   1. wait for an engine build already running (WAIT_PID), if any
#   2. build any missing TensorRT engines (existing ones are reused)
#   3. dry run (benchmark.py measure --quick) on one model, and check it: every
#      configuration needs a valid repeat with a positive latency and energy
#   4. full measurement of every model
# Stops at the first failure; the last log line says which step. Each step's output is
# in run_scripts/results/gpu_baseline/ (see benchmark.py).
#
#   PY=python CPUS=8-11 DRY_MODEL=resnet50 nohup run_scripts/gpu_baseline/run_all.sh > run_all.log 2>&1 &
set -u
cd "$(dirname "$0")/../.."
: "${PY:?set PY to the Python of the GPU environment (requirements.txt)}"
: "${CPUS:?set CPUS to the CPUs to pin the benchmark to, e.g. 8-11}"
: "${DRY_MODEL:?set DRY_MODEL to the model of the dry run, e.g. resnet50}"
log() { echo "[$(date +%H:%M:%S)] $*"; }

if [ -n "${WAIT_PID:-}" ]; then
    log "waiting for running build (pid $WAIT_PID)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do sleep 10; done
fi

log "STEP build"
$PY run_scripts/gpu_baseline/benchmark.py build --cpus "$CPUS" || { log "PIPELINE FAILED: build"; exit 1; }

log "STEP dry run ($DRY_MODEL)"
$PY run_scripts/gpu_baseline/benchmark.py measure --quick --models "$DRY_MODEL" --cpus "$CPUS" \
    || { log "PIPELINE FAILED: dry run"; exit 2; }
DRY_MODEL="$DRY_MODEL" $PY - <<'EOF' || { log "PIPELINE FAILED: dry-run check"; exit 3; }
import json, os, socket
path = f"run_scripts/results/gpu_baseline/benchmark_{socket.gethostname()}_quick.json"
runs = json.load(open(path))["models"][os.environ["DRY_MODEL"]]
for cfg, r in runs.items():
    s = r["summary"]
    assert s.get("valid_repeats", 0) >= 1, f"{cfg}: no valid repeat"
    assert s["latency_ms_median"] > 0 and s["mJ_per_inference"] > 0, f"{cfg}: {s}"
print("dry run ok:", {c: (r["summary"]["latency_ms_median"], r["summary"]["mJ_per_inference"]) for c, r in runs.items()})
EOF

log "STEP full measurement"
$PY run_scripts/gpu_baseline/benchmark.py measure --cpus "$CPUS" || { log "PIPELINE FAILED: measurement"; exit 4; }
log "PIPELINE DONE"
