"""Run NeuroSim's inference-accuracy simulation for one 1F1C configuration.

Reads <config_dir>/neurosim.json, passes every key under "args" to
NeuroSim/Inference_pytorch/inference.py as --key value, using NeuroSim/.venv.
Output goes to <config_dir>/neurosim_out/<timestamp>/:
  stdout.log   full console output (Python accuracy run + NeuroSim C++ hardware run)
  result.json  the settings used and the test accuracy

inference.py always ends by running NeuroSim's C++ hardware estimate (Param.cpp:
a generic 22 nm SRAM chip, not this macro). Its numbers are kept in stdout.log
but are not used; energy/latency/area come from the SPICE and NVSim models.

CIFAR-10 is downloaded by NeuroSim to /tmp/public_dataset/pytorch on first use.

Usage: python3 targets/1fefet1c/run_neurosim.py <config_dir>   (e.g. default)
"""
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

HERE = Path(__file__).resolve().parent
NEUROSIM = HERE / "NeuroSim" / "Inference_pytorch"
PYTHON = HERE / "NeuroSim" / ".venv" / "bin" / "python"


def main():
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    cfg_dir = (HERE / sys.argv[1]).resolve()
    cfg = json.loads((cfg_dir / "neurosim.json").read_text())
    args = cfg["args"]

    out_dir = cfg_dir / "neurosim_out" / datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir.mkdir(parents=True)
    cmd = [str(PYTHON), "inference.py", "--logdir", str(out_dir / "neurosim_log")]
    for k, v in args.items():
        cmd += [f"--{k}", str(v)]
    print("running:", " ".join(cmd), flush=True)

    log_path = out_dir / "stdout.log"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=NEUROSIM, stdout=subprocess.PIPE,
                                stderr=subprocess.STDOUT, text=True)
        for line in proc.stdout:
            log.write(line)
            if "Accuracy" in line or "Error" in line or "Traceback" in line:
                print(line, end="", flush=True)
        proc.wait()

    hit = re.search(r"Accuracy: (\d+)/(\d+)", log_path.read_text())
    result = {"config": sys.argv[1], "args": args, "returncode": proc.returncode,
              "correct": int(hit.group(1)) if hit else None,
              "total": int(hit.group(2)) if hit else None,
              "accuracy_pct": 100 * int(hit.group(1)) / int(hit.group(2)) if hit else None}
    (out_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    if hit:
        # also into <config>/results/accuracy.json, keyed by the variation used
        os.environ["F1C_CONFIG"] = sys.argv[1]
        from f1c_io import LALENI, save_result
        save_result("accuracy", f"vari_{args.get('vari', 0)}", {
            "accuracy_pct": result["accuracy_pct"], "correct": result["correct"], "total": result["total"],
            "neurosim_args": args, "run_dir": str(out_dir.relative_to(HERE)),
            "laleni_accuracy_pct": LALENI["cifar10_vgg8_accuracy_pct"]})
        print(f"accuracy: {result['accuracy_pct']:.2f}%  (Laleni: 89%)  -> {out_dir}")
    else:
        print(f"no accuracy found, see {log_path}")


if __name__ == "__main__":
    main()
