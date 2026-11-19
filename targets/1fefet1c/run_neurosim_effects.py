"""NeuroSim's VGG8 / CIFAR-10 inference with the 1F1C macro's nonidealities
(NeuroSim/Inference_pytorch/utee/f1c_effects.py), one at a time and together, to see how
far each moves the accuracy from our ideal-macro run (91.80 %) toward Laleni et al.'s 89 %.

Every variant takes <config_dir>/neurosim.json's "args" with vari = 0 (the f1c switches
model the Vth spread instead), plus its f1c_* switches:

  baseline     none (reproduces the vari_0.0 run)
  levels_main  Sec. V's Vth set [1.2, 0.9, 0.6, 0.3] V: cell levels VDD - Vth (eq. 1)
  vth_sigma    Fig. 7(b)'s per-state Vth spread
  fixed_adc    the ADC with a fixed 0.5 V full scale, charge-sharing gain kappa = 0.69
  enob         the ADC at 6.4 effective bits
  all_main     all four, Sec. V's Vth set
  all_linear   all four, Fig. 10(b)'s linear Vth set [1.2, 0.7, 0.4, 0.1] V

Runs with NeuroSim/.venv (setup_neurosim.sh), on --device cuda, mps or cpu (default: CUDA
if available, else the CPU). Each run's console output goes to
<config_dir>/neurosim_out/<time>_<variant>/; the accuracies go to
<config_dir>/results/accuracy_effects.json, resumable per variant.

Usage: python3 targets/1fefet1c/run_neurosim_effects.py default [--device mps] [variant ...]
"""
import json
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
NEUROSIM = HERE / "NeuroSim" / "Inference_pytorch"
PYTHON = HERE / "NeuroSim" / ".venv" / "bin" / "python"
LALENI_ACCURACY_PCT = 89

ALL_ON = {"f1c_vth_sigma": 1, "f1c_fixed_adc": 1, "f1c_kappa": 0.69, "f1c_enob": 6.4}
VARIANTS = {
    "baseline": {},
    "levels_main": {"f1c_levels": "main"},
    "vth_sigma": {"f1c_vth_sigma": 1},
    "fixed_adc": {"f1c_fixed_adc": 1, "f1c_kappa": 0.69},
    "enob": {"f1c_enob": 6.4},
    "all_main": {"f1c_levels": "main", **ALL_ON},
    "all_linear": {"f1c_levels": "linear", **ALL_ON},
}


def run(cfg_dir: Path, name: str, args: dict) -> dict:
    out_dir = cfg_dir / "neurosim_out" / f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{name}"
    out_dir.mkdir(parents=True)
    cmd = [str(PYTHON), "inference.py", "--logdir", str(out_dir / "neurosim_log")]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    print(f"[{name}] running: {' '.join(cmd)}", flush=True)
    log_path = out_dir / "stdout.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=NEUROSIM, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if "Accuracy" in line or "Error" in line or "Traceback" in line:
                print(f"[{name}] {line}", end="", flush=True)
        proc.wait()
    hit = re.search(r"Accuracy: (\d+)/(\d+)", log_path.read_text())
    if not hit:
        raise RuntimeError(f"[{name}] no accuracy in {log_path} (exit {proc.returncode})")
    correct, total = int(hit.group(1)), int(hit.group(2))
    return {"accuracy_pct": 100 * correct / total, "correct": correct, "total": total,
            "neurosim_args": args, "run_dir": str(out_dir.relative_to(HERE)),
            "laleni_accuracy_pct": LALENI_ACCURACY_PCT}


def main() -> None:
    argv = sys.argv[1:]
    device = None
    if "--device" in argv:
        i = argv.index("--device")
        device = argv[i + 1]
        del argv[i:i + 2]
    if not argv:
        sys.exit(__doc__)
    cfg_dir = (HERE / argv[0]).resolve()
    names = argv[1:] or list(VARIANTS)
    unknown = [n for n in names if n not in VARIANTS]
    if unknown:
        sys.exit(f"unknown variants {unknown}; choose from {list(VARIANTS)}")
    base = {**json.loads((cfg_dir / "neurosim.json").read_text())["args"], "vari": 0.0}
    out_path = cfg_dir / "results" / "accuracy_effects.json"
    results = json.loads(out_path.read_text()) if out_path.exists() else {}
    for name in names:
        args = {**base, **VARIANTS[name]}
        if name in results and results[name].get("neurosim_args") == args:
            print(f"[{name}] already done: {results[name]['accuracy_pct']:.2f}%", flush=True)
            continue
        run_args = {**args, "device": device} if device else args
        results[name] = {"generated": datetime.now().isoformat(timespec="seconds"), **run(cfg_dir, name, run_args)}
        results[name]["neurosim_args"] = args  # without the device: the accuracy does not depend on it
        out_path.write_text(json.dumps(results, indent=1) + "\n")
        print(f"[{name}] accuracy {results[name]['accuracy_pct']:.2f}%  (Laleni: {LALENI_ACCURACY_PCT}%)", flush=True)
    print("\n".join(f"{n:12s} {r['accuracy_pct']:6.2f}%" for n, r in results.items()))


if __name__ == "__main__":
    main()
