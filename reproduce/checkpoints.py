"""The int8 models the compiler maps (dump/models/, not in the repository): one build step
per model of the mapping latency figure, each with the arguments that made the paper's
checkpoint (every rebuild is identical to it). The -w3 kernels derive from these. A step
is skipped if its checkpoint exists. --force leaves them: several scripts share them, and
a rebuild gives the same file.
"""
import sys

from common import REPO, Step, file_exists

PY = sys.executable
MODELS_DIR = REPO / "dump" / "models"

# model (kernels/registry.py) -> (checkpoint, run_scripts/measure_accuracy build module, its arguments)
CHECKPOINTS = {
    "resnet50": ("resnet50_int8.pt", "build_resnet50_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "mobilenet-v2": ("mobilenet_v2_int8.pt", "build_mobilenet_v2_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "vgg16": ("vgg16_int8.pt", "build_vgg16_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "vit-base": ("vit_base_int8.pt", "build_vit_base_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "vit-large": ("vit_large_int8.pt", "build_vit_large_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "bert-base": ("bert_base_int8.pt", "build_bert_base_int8",
                  ["--n-calib", "32", "--calib-seed", "1", "--smooth", "qkv_fc1_fc2"]),
    "bert-large": ("bert_large_int8.pt", "build_bert_large_int8", ["--n-calib", "64", "--calib-seed", "1"]),
    "gpt2": ("gpt2_int8.pt", "build_gpt2_int8", ["--n-calib", "16", "--calib-seed", "1", "--n-eval", "16"]),
}


def checkpoint_steps(models) -> list[Step]:
    """A build step for each of `models`' int8 checkpoints."""
    steps = []
    for model in models:
        name, module, args = CHECKPOINTS[model]
        steps.append(Step(f"build int8 {model}", [PY, "-m", f"run_scripts.measure_accuracy.{module}", *args],
                          REPO, file_exists(MODELS_DIR / name), {}, []))
    return steps
