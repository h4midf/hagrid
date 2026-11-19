"""Builds the int8 vit-base that the compiler uses: real google/vit-base-patch16-224
weights, quantized with kernels/vit.py's scheme (quantize_vit_encoder), calibrated on
real ImageNet validation images, saved to kernels.vit_base.INT8_CHECKPOINT.

Before quantizing, the float encoder + classifier is checked against Hugging Face's
logits. Reports top-1 of the float and the int8 model on the calibration images.

Usage (from the repo root):
  python3 -m run_scripts.measure_accuracy.build_vit_base_int8 --n-calib 64 --calib-seed 1
"""
import argparse
import os

import torch

from kernels.vit import quantize_vit_encoder
from kernels.vit_base import INT8_CHECKPOINT, ViTEncoder
from run_scripts.measure_accuracy.evaluate_vit_large_variant_configs import (
    VIT_VARIANTS, load_images, load_real_weights, real_classify, real_patch_embed,
)

MAX_LOGIT_DIFF = 1e-3   # float encoder vs Hugging Face, in double


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-calib", type=int, required=True, help="real ImageNet images the observers see")
    parser.add_argument("--calib-seed", type=int, required=True, help="shuffle seed of the calibration images")
    args = parser.parse_args()

    model = ViTEncoder().eval().double()
    hf_model, classifier = load_real_weights(model, VIT_VARIANTS["vit-base"])
    images, labels = load_images(args.n_calib, seed=args.calib_seed)

    with torch.no_grad():
        mine = real_classify(classifier, model(real_patch_embed(hf_model, images[0].unsqueeze(0))))[0]
        ref = hf_model(pixel_values=images[0].unsqueeze(0).double()).logits[0]
    max_diff = float((mine - ref).abs().max())
    print(f"check vs HF logits: max abs diff = {max_diff}")
    if max_diff > MAX_LOGIT_DIFF:
        raise RuntimeError(f"ViTEncoder + classifier does not match Hugging Face vit-base (diff {max_diff})")

    model = model.float()
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
