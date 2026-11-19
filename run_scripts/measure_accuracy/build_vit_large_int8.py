"""Builds the int8 vit-large that the compiler uses: real google/vit-large-patch16-224
weights, quantized with kernels/vit.py's scheme (quantize_vit_encoder), calibrated on
real ImageNet validation images, saved to kernels.vit.INT8_CHECKPOINT.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_vit_large_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import os

import torch

from kernels.vit import INT8_CHECKPOINT, ViTEncoder, quantize_vit_encoder
from run_scripts.measure_accuracy.evaluate_vit_large_variant_configs import (
    VIT_VARIANTS, load_images, load_real_weights, real_classify, real_patch_embed,
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real ImageNet images the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration images")
    args = parser.parse_args()

    model = ViTEncoder().eval()
    hf_model, classifier = load_real_weights(model, VIT_VARIANTS["vit-large"])
    images, labels = load_images(args.n_calib, seed=args.calib_seed)
    embeddings = [real_patch_embed(hf_model, image.unsqueeze(0)).float() for image in images]

    quantized = quantize_vit_encoder(model, embeddings)

    with torch.no_grad():
        float_correct = sum(
            int(torch.argmax(real_classify(classifier, model(e).double())[0]).item() == label)
            for e, label in zip(embeddings, labels)
        )
        int8_correct = sum(
            int(torch.argmax(real_classify(classifier, quantized(e).double())[0]).item() == label)
            for e, label in zip(embeddings, labels)
        )
    print(f"on the {len(labels)} calibration images: float top-1 {float_correct}/{len(labels)}, "
          f"int8 top-1 {int8_correct}/{len(labels)}")

    os.makedirs(os.path.dirname(INT8_CHECKPOINT), exist_ok=True)
    torch.save(quantized, INT8_CHECKPOINT)
    print(f"wrote {INT8_CHECKPOINT}")


if __name__ == "__main__":
    main()
