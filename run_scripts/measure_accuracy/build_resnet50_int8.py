"""Builds the int8 resnet50 that the compiler uses: real torchvision ResNet-50 v1.5
weights (ResNet50_Weights.IMAGENET1K_V1), quantized with kernels/resnet50.py's scheme
(quantize_resnet50), calibrated on real ImageNet validation images, saved to
kernels.resnet50.INT8_CHECKPOINT.

Images: the same ImageNet-1k mirror as the ViT evaluation, with torchvision's own
preprocessing for these weights (resize 256, center crop 224, ImageNet mean/std).
Before quantizing, our kernel with the mapped weights is checked against torchvision's
own model. Reports top-1 of the float and the int8 model on the calibration images.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_resnet50_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import itertools
import os

import torch
from datasets import load_dataset
from torchvision.models import ResNet50_Weights, resnet50

from kernels.resnet50 import INT8_CHECKPOINT, ResNet50Int8, ResNet50Kernel, quantize_resnet50
from run_scripts.measure_accuracy.evaluate_vit_large_variant_configs import (
    VIT_DATASET_NAME, VIT_IMAGE_COLUMN, VIT_SOURCE_SPLIT,
)

WEIGHTS = ResNet50_Weights.IMAGENET1K_V1
MAX_LOGIT_DIFF = 1e-3   # our kernel vs torchvision, float32


def _our_name(name: str) -> str:
    """torchvision's parameter name -> kernels/resnet50.py's (stem and stage names)."""
    for theirs, ours in (("conv1.", "stem_conv."), ("bn1.", "stem_bn.")):
        if name.startswith(theirs):
            return ours + name[len(theirs):]
    if name.startswith("layer"):
        return "stage" + name[len("layer"):]
    return name


def load_real_weights(kernel: ResNet50Kernel):
    """torchvision's ResNet-50 v1.5 weights into `kernel` (strict); returns torchvision's model."""
    tv = resnet50(weights=WEIGHTS).eval()
    kernel.load_state_dict({_our_name(k): v for k, v in tv.state_dict().items()}, strict=True)
    return tv


def load_images(n: int, seed: int, weights):
    """n shuffled ImageNet validation images with torchvision's preprocessing for
    `weights` (a torchvision weights enum)."""
    preprocess = weights.transforms()
    stream = load_dataset(VIT_DATASET_NAME, split=VIT_SOURCE_SPLIT, streaming=True)
    pool = list(itertools.islice(stream.shuffle(seed=seed, buffer_size=max(n * 4, 1000)), n))
    if len(pool) < n:
        raise RuntimeError(f"{VIT_DATASET_NAME}/{VIT_SOURCE_SPLIT} only yielded {len(pool)} images, need {n}")
    images = [preprocess(ex[VIT_IMAGE_COLUMN].convert("RGB")).unsqueeze(0) for ex in pool]
    return images, [int(ex["label"]) for ex in pool]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real ImageNet images the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration images")
    args = parser.parse_args()

    model = ResNet50Kernel().eval()
    tv = load_real_weights(model)
    images, labels = load_images(args.n_calib, args.calib_seed, WEIGHTS)

    with torch.no_grad():
        max_diff = float((model(images[0]) - tv(images[0])).abs().max())
    print(f"check vs torchvision logits: max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"ResNet50Kernel does not match torchvision's resnet50 (diff {max_diff})")

    quantized = quantize_resnet50(model, images)

    with torch.no_grad():
        float_correct = sum(int(torch.argmax(model(x)[0]).item() == y) for x, y in zip(images, labels))
        int8_correct = sum(int(torch.argmax(quantized(x)[0]).item() == y) for x, y in zip(images, labels))
    print(f"on the {len(labels)} calibration images: float top-1 {float_correct}/{len(labels)}, "
          f"int8 top-1 {int8_correct}/{len(labels)}")

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized.state_dict(), INT8_CHECKPOINT)
    with torch.no_grad():
        reload_diff = max(float((ResNet50Int8()(x) - quantized(x)).abs().max()) for x in images[:8])
    if reload_diff != 0.0:
        raise RuntimeError(f"ResNet50Int8 reloaded from {INT8_CHECKPOINT} differs from the built model ({reload_diff})")
    print(f"wrote {INT8_CHECKPOINT} (reloads exactly)")


if __name__ == "__main__":
    main()
