"""ViT's Hugging Face weights and ImageNet-1k images, for the build_*_int8.py scripts
(vit-base, vit-large, and resnet50's dataset).

The kernel (kernels/vit.py) starts after the patch embedding and ends at the final norm:
real_patch_embed and real_classify (the classifier on the CLS token, as
ViTForImageClassification) run outside it. The checkpoints are the 1000-class
google/vit-*-patch16-224 (not the -in21k pretraining ones).
"""
import itertools

import numpy as np
import torch
import torch.nn as nn
from datasets import load_dataset
from transformers import AutoModelForImageClassification

VIT_VARIANTS = {
    "vit-base": "google/vit-base-patch16-224",
    "vit-large": "google/vit-large-patch16-224",
}
# ImageNet-1k validation images from an ungated mirror with a "label" column (the official
# dataset is gated); only a "val" split exists
VIT_DATASET_NAME = "evanarlian/imagenet_1k_resized_256"
VIT_SOURCE_SPLIT = "val"
VIT_IMAGE_COLUMN = "image"
VIT_IMAGE_SIZE = (224, 224)
VIT_IMAGE_MEAN = (0.5, 0.5, 0.5)
VIT_IMAGE_STD = (0.5, 0.5, 0.5)


def _preprocess_vit_image(image):
    image = image.convert("RGB").resize(VIT_IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
    mean = torch.tensor(VIT_IMAGE_MEAN, dtype=torch.float32).view(3, 1, 1)
    std = torch.tensor(VIT_IMAGE_STD, dtype=torch.float32).view(3, 1, 1)
    return (tensor - mean) / std


SEED = 0


def load_real_weights(kernel: nn.Module, checkpoint: str) -> tuple:
    """A Hugging Face ViT checkpoint (a VIT_VARIANTS value) loaded onto `kernel`'s
    parameter names, for every layer it has: attention.attention.{query,key,value} ->
    attn.{Wq,Wk,Wv}, attention.output.dense -> attn.proj, intermediate.dense -> mlp.fc1,
    output.dense -> mlp.fc2, layernorm_{before,after} -> norm{1,2}. Returns (the Hugging
    Face model, the classifier)."""
    hf = AutoModelForImageClassification.from_pretrained(checkpoint).eval().double()
    sd = hf.state_dict()

    def g(name):
        return sd[name].detach().clone()

    sdict = kernel.state_dict()
    for l in range(len(kernel.layers)):
        p = f"vit.encoder.layer.{l}."
        sdict[f"layers.{l}.attn.Wq.weight"] = g(p + "attention.attention.query.weight")
        sdict[f"layers.{l}.attn.Wq.bias"] = g(p + "attention.attention.query.bias")
        sdict[f"layers.{l}.attn.Wk.weight"] = g(p + "attention.attention.key.weight")
        sdict[f"layers.{l}.attn.Wk.bias"] = g(p + "attention.attention.key.bias")
        sdict[f"layers.{l}.attn.Wv.weight"] = g(p + "attention.attention.value.weight")
        sdict[f"layers.{l}.attn.Wv.bias"] = g(p + "attention.attention.value.bias")
        sdict[f"layers.{l}.attn.proj.weight"] = g(p + "attention.output.dense.weight")
        sdict[f"layers.{l}.attn.proj.bias"] = g(p + "attention.output.dense.bias")
        sdict[f"layers.{l}.norm1.weight"] = g(p + "layernorm_before.weight")
        sdict[f"layers.{l}.norm1.bias"] = g(p + "layernorm_before.bias")
        sdict[f"layers.{l}.mlp.fc1.weight"] = g(p + "intermediate.dense.weight")
        sdict[f"layers.{l}.mlp.fc1.bias"] = g(p + "intermediate.dense.bias")
        sdict[f"layers.{l}.mlp.fc2.weight"] = g(p + "output.dense.weight")
        sdict[f"layers.{l}.mlp.fc2.bias"] = g(p + "output.dense.bias")
        sdict[f"layers.{l}.norm2.weight"] = g(p + "layernorm_after.weight")
        sdict[f"layers.{l}.norm2.bias"] = g(p + "layernorm_after.bias")
    sdict["final_norm.weight"] = g("vit.layernorm.weight")
    sdict["final_norm.bias"] = g("vit.layernorm.bias")
    kernel.load_state_dict(sdict)

    out_features, in_features = sd["classifier.weight"].shape
    classifier = nn.Linear(int(in_features), int(out_features)).eval().double()
    classifier.load_state_dict({"weight": g("classifier.weight"), "bias": g("classifier.bias")})
    return hf, classifier


def real_patch_embed(hf_model, pixel_values: torch.Tensor) -> torch.Tensor:
    """ViT's patch embedding, which the kernel does not include. pixel_values:
    (1, 3, 224, 224)."""
    emb = hf_model.vit.embeddings
    with torch.no_grad():
        return emb(pixel_values.double())


def real_classify(classifier: nn.Module, hidden: torch.Tensor) -> torch.Tensor:
    """ViTForImageClassification's head: the classifier on the CLS token (position 0)."""
    with torch.no_grad():
        return classifier(hidden[:, 0, :])


def load_images(n: int, seed: int = SEED):
    """n preprocessed images and their labels from VIT_DATASET_NAME, streamed and
    shuffled (the val split is ordered by class)."""
    stream = load_dataset(VIT_DATASET_NAME, split=VIT_SOURCE_SPLIT, streaming=True)
    stream = stream.shuffle(seed=seed, buffer_size=max(n * 4, 1000))
    pool = list(itertools.islice(stream, n))
    if len(pool) < n:
        raise RuntimeError(f"{VIT_DATASET_NAME}/{VIT_SOURCE_SPLIT} only yielded {len(pool)} images, need {n}")
    images = [_preprocess_vit_image(ex[VIT_IMAGE_COLUMN]) for ex in pool]
    labels = [int(ex["label"]) for ex in pool]
    return images, labels
