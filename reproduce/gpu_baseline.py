"""Reproduces one GPU reference line of the paper's mapping latency and energy figures:
TensorRT INT8 latency and whole-board energy per inference of every model, batch 1, on
the GPU this runs on. Runs only on a machine with an NVIDIA GPU and the environment of
run_scripts/gpu_baseline/requirements.txt.

  1. run_scripts/gpu_baseline/run_all.sh: builds the TensorRT engines, a dry run, then the
     full measurement (benchmark.py), into run_scripts/results/gpu_baseline/
     benchmark_<host>.json
  2. run_scripts/gpu_baseline/summarize.py: its summary.md
  3. writes dump/gpu_baseline/latency_<device>.json, the file the figures read: per model
     the TensorRT INT8 median latency (latency_ns) and every configuration's summary

Skipped when latency_<device>.json exists; --force measures again. Then compares the new
latency with the figure's (plot/meta_data/mapping_latency.json) and writes
reproduce/results/gpu_<device>.csv. A measurement on real hardware is not exactly
repeatable: the CSV gives the ratio to the figure; the run fails only if a model is
missing.

Usage (on the GPU machine, from the repo root):
  python reproduce/gpu_baseline.py --device rtx3090 --label "NVIDIA GeForce RTX 3090" --cpus 8-11 \
      --dry-model resnet50 [--force]
"""
import argparse
import json
import platform
import socket
import shutil
import subprocess
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import REPO, RESULTS, Step, file_exists, run_steps, write_csv  # noqa: E402
from mapping_common import MODELS, TARGETS, figure_rows  # noqa: E402

GPU_DIR = REPO / "run_scripts" / "gpu_baseline"
GPU_RESULTS = REPO / "run_scripts" / "results" / "gpu_baseline"
OUT_DIR = REPO / "dump" / "gpu_baseline"
# --device -> the key of its reference line in the figures' saved data
FIGURE_KEY = {"rtx3090": "gpu", "rtx3060_laptop": "gpu_mobile"}
CONFIG = "trt_int8"


def latency_file(device: str) -> Path:
    return OUT_DIR / f"latency_{device}.json"


def write_latency(device: str, label: str) -> None:
    """dump/gpu_baseline/latency_<device>.json from this host's benchmark_<host>.json."""
    host = socket.gethostname()
    source = GPU_RESULTS / f"benchmark_{host}.json"
    bench = json.loads(source.read_text())
    models = bench["models"]
    missing = [m for m in MODELS if m not in models or models[m][CONFIG]["summary"]["valid_repeats"] == 0]
    if missing:
        raise RuntimeError(f"{source}: no valid {CONFIG} measurement for {missing}")
    env = bench["environment"]
    kernels_commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, capture_output=True,
                                    text=True, check=True).stdout.strip()
    record = {
        "source_path": str(source.relative_to(REPO)),
        "device": f"{label} ({env['power_limit_W']:.0f} W power limit)",
        "environment": {"torch": env["torch"], "tensorrt": env["tensorrt"], "python": platform.python_version(),
                        "kernels_commit": kernels_commit, "host": host, "measured": env["date"]},
        "configuration": CONFIG,
        "batch_size": 1,
        "latency_ns": {m: models[m][CONFIG]["summary"]["latency_ms_median"] * 1e6 for m in MODELS},
        "measurement_summaries": {m: {c: r["summary"] for c, r in models[m].items()} for m in MODELS},
    }
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    latency_file(device).write_text(json.dumps(record, indent=1, allow_nan=False) + "\n")


def compare(device: str) -> list[dict]:
    record = json.loads(latency_file(device).read_text())
    figure = figure_rows()[TARGETS[0]]   # every target's rows carry the same GPU line
    rows = []
    for model in MODELS:
        ours = record["latency_ns"][model]
        paper = figure[model][0][FIGURE_KEY[device]]["latency_ns"]
        rows.append({"device": device, "model": model, "latency_ns": ours, "figure_latency_ns": paper,
                     "ratio": f"{ours / paper:.3f}"})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--device", required=True, choices=sorted(FIGURE_KEY),
                        help="which of the figures' GPU lines this GPU measures")
    parser.add_argument("--label", required=True, help="the GPU's name, as the figures' provenance gives it")
    parser.add_argument("--cpus", required=True, help="CPUs to pin the benchmark to, e.g. 8-11")
    parser.add_argument("--dry-model", required=True, choices=MODELS, help="the model of run_all.sh's dry run")
    parser.add_argument("--force", action="store_true", help="measure again")
    args = parser.parse_args()
    if shutil.which("nvidia-smi") is None:
        raise SystemExit("no NVIDIA GPU here (nvidia-smi not found): run this on the GPU machine")

    env = {"PY": sys.executable, "CPUS": args.cpus, "DRY_MODEL": args.dry_model}
    log = run_steps([Step(f"GPU {args.device}: TensorRT build and measurement", ["bash", str(GPU_DIR / "run_all.sh")],
                          REPO, file_exists(latency_file(args.device)), env, [latency_file(args.device)])],
                    args.force)
    if log[0]["status"] == "ran":
        subprocess.run([sys.executable, str(GPU_DIR / "summarize.py"), "--host", socket.gethostname()],
                       cwd=REPO, check=True)
        write_latency(args.device, args.label)
    rows = compare(args.device)
    write_csv(RESULTS / f"gpu_{args.device}.csv", rows)
    for r in rows:
        print(f"  {r['model']:13s} {r['latency_ns'] / 1e6:9.3f} ms (figure {r['figure_latency_ns'] / 1e6:9.3f} ms), "
              f"ratio {r['ratio']}")
    print(f"wrote {RESULTS / f'gpu_{args.device}.csv'}")


if __name__ == "__main__":
    main()
