"""Builds the int8 vgg16 that the compiler uses: real torchvision VGG-16 weights
(VGG16_Weights.IMAGENET1K_V1), quantized with kernels/vgg16.py's scheme
(quantize_vgg16), calibrated on real ImageNet validation images, saved to
kernels.vgg16.INT8_CHECKPOINT.

Images: the same ImageNet-1k mirror and loader as build_resnet50_int8.py, with
torchvision's preprocessing for these weights. Before quantizing, our kernel with the
mapped weights is checked against torchvision's own model (which also checks that the
average pool and dropouts our kernel leaves out are the identity). Reports top-1 of the
float and the int8 model on the calibration images.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_vgg16_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import os

import torch
from torchvision.models import VGG16_Weights, vgg16

from kernels.vgg16 import INT8_CHECKPOINT, VGG16Int8, VGG16Kernel, quantize_vgg16
from run_scripts.measure_accuracy.build_resnet50_int8 import load_images

WEIGHTS = VGG16_Weights.IMAGENET1K_V1
MAX_LOGIT_DIFF = 1e-3   # our kernel vs torchvision, float32
# torchvision's classifier Sequential (0 Linear, 1 ReLU, 2 Dropout, 3 Linear, ...) -> ours
_CLASSIFIER = {"classifier.0.": "fc1.", "classifier.3.": "fc2.", "classifier.6.": "fc3."}


def _our_name(name: str) -> str:
    """torchvision's parameter name -> kernels/vgg16.py's."""
    for theirs, ours in _CLASSIFIER.items():
        if name.startswith(theirs):
            return ours + name[len(theirs):]
    return name


def load_real_weights(kernel: VGG16Kernel):
    """torchvision's VGG-16 weights into `kernel` (strict); returns torchvision's model."""
    tv = vgg16(weights=WEIGHTS).eval()
    kernel.load_state_dict({_our_name(k): v for k, v in tv.state_dict().items()}, strict=True)
    return tv


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real ImageNet images the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration images")
    args = parser.parse_args()

    model = VGG16Kernel().eval()
    tv = load_real_weights(model)
    images, labels = load_images(args.n_calib, args.calib_seed, WEIGHTS)

    with torch.no_grad():
        max_diff = float((model(images[0]) - tv(images[0])).abs().max())
    print(f"check vs torchvision logits: max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"VGG16Kernel does not match torchvision's vgg16 (diff {max_diff})")

    quantized = quantize_vgg16(model, images)

    def top1(m):
        with torch.no_grad():
            return sum(int(torch.argmax(m(x)[0]).item() == y) for x, y in zip(images, labels))

    n = len(labels)
    print(f"on the {n} calibration images: float top-1 {top1(model)}/{n}, int8 top-1 {top1(quantized)}/{n}")

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized.state_dict(), INT8_CHECKPOINT)
    with torch.no_grad():
        reload_diff = max(float((VGG16Int8()(x) - quantized(x)).abs().max()) for x in images[:8])
    if reload_diff != 0.0:
        raise RuntimeError(f"VGG16Int8 reloaded from {INT8_CHECKPOINT} differs from the built model ({reload_diff})")
    print(f"wrote {INT8_CHECKPOINT} (reloads exactly)")


if __name__ == "__main__":
    main()
