"""Builds the int8 mobilenet-v2 that the compiler uses: real torchvision MobileNet-v2
weights (MobileNet_V2_Weights.IMAGENET1K_V1), with ReLU in place of ReLU6 (see
kernels/mobilenet_v2.py), quantized with quantize_mobilenet_v2, calibrated on real
ImageNet validation images, saved to kernels.mobilenet_v2.INT8_CHECKPOINT.

Images: the same ImageNet-1k mirror and loader as build_resnet50_int8.py, with
torchvision's preprocessing for these weights. Before quantizing, our kernel with the
mapped weights is checked against torchvision's own model with the same ReLU6 -> ReLU
swap. Reports top-1 of torchvision's original (ReLU6) model, our float model and the
int8 model on the calibration images.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_mobilenet_v2_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import os

import torch
import torch.nn as nn
from torchvision.models import MobileNet_V2_Weights, mobilenet_v2

from kernels.mobilenet_v2 import INT8_CHECKPOINT, MobileNetV2Int8, MobileNetV2Kernel, quantize_mobilenet_v2
from run_scripts.measure_accuracy.build_resnet50_int8 import load_images

WEIGHTS = MobileNet_V2_Weights.IMAGENET1K_V1
MAX_LOGIT_DIFF = 1e-3   # our kernel vs torchvision with ReLU, float32


def _our_name(name: str) -> str:
    """torchvision's parameter name -> kernels/mobilenet_v2.py's (its Linear is classifier.1)."""
    if name.startswith("classifier.1."):
        return "classifier." + name[len("classifier.1."):]
    return name


def _relu6_to_relu(module: nn.Module) -> None:
    for name, child in module.named_children():
        if isinstance(child, nn.ReLU6):
            setattr(module, name, nn.ReLU(inplace=False))
        else:
            _relu6_to_relu(child)


def load_real_weights(kernel: MobileNetV2Kernel):
    """torchvision's MobileNet-v2 weights into `kernel` (strict). Returns torchvision's
    model as published (ReLU6) and the same model with ReLU, both in eval mode."""
    tv = mobilenet_v2(weights=WEIGHTS).eval()
    kernel.load_state_dict({_our_name(k): v for k, v in tv.state_dict().items()}, strict=True)
    tv_relu = mobilenet_v2(weights=WEIGHTS)
    _relu6_to_relu(tv_relu)
    return tv, tv_relu.eval()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real ImageNet images the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration images")
    args = parser.parse_args()

    model = MobileNetV2Kernel().eval()
    tv, tv_relu = load_real_weights(model)
    images, labels = load_images(args.n_calib, args.calib_seed, WEIGHTS)

    with torch.no_grad():
        max_diff = float((model(images[0]) - tv_relu(images[0])).abs().max())
    print(f"check vs torchvision (ReLU) logits: max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"MobileNetV2Kernel does not match torchvision's mobilenet_v2 with ReLU (diff {max_diff})")

    quantized = quantize_mobilenet_v2(model, images)

    def top1(m):
        with torch.no_grad():
            return sum(int(torch.argmax(m(x)[0]).item() == y) for x, y in zip(images, labels))

    n = len(labels)
    print(f"on the {n} calibration images: torchvision (ReLU6) top-1 {top1(tv)}/{n}, "
          f"float (ReLU) {top1(model)}/{n}, int8 {top1(quantized)}/{n}")

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized.state_dict(), INT8_CHECKPOINT)
    with torch.no_grad():
        reload_diff = max(float((MobileNetV2Int8()(x) - quantized(x)).abs().max()) for x in images[:8])
    if reload_diff != 0.0:
        raise RuntimeError(f"MobileNetV2Int8 reloaded from {INT8_CHECKPOINT} differs from the built model ({reload_diff})")
    print(f"wrote {INT8_CHECKPOINT} (reloads exactly)")


if __name__ == "__main__":
    main()
