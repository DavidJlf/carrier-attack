#!/usr/bin/env python3
"""Evaluate one fixed white-box image on black-box ImageNet classifiers.

The image is never regenerated or modified. Each classifier uses the official
preprocessing bundled with its own torchvision weights.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import (
    ConvNeXt_Base_Weights,
    Inception_V3_Weights,
    ResNet101_Weights,
    Swin_B_Weights,
    VGG19_Weights,
    convnext_base,
    inception_v3,
    resnet101,
    swin_b,
    vgg19,
)


MODEL_SPECS = {
    "resnet101": (resnet101, ResNet101_Weights.DEFAULT),
    "vgg19": (vgg19, VGG19_Weights.DEFAULT),
    "inception_v3": (inception_v3, Inception_V3_Weights.DEFAULT),
    "convnext_base": (convnext_base, ConvNeXt_Base_Weights.DEFAULT),
    "swin_b": (swin_b, Swin_B_Weights.DEFAULT),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run black-box inference on an existing attacked image."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-class", required=True, type=int)
    parser.add_argument(
        "--models", nargs="+", choices=tuple(MODEL_SPECS),
        default=list(MODEL_SPECS),
    )
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but CUDA is unavailable")
    return torch.device(requested)


def evaluate_model(
    name: str,
    image: Image.Image,
    target_class: int,
    device: torch.device,
) -> dict:
    constructor, weights = MODEL_SPECS[name]
    categories = weights.meta["categories"]
    if not 0 <= target_class < len(categories):
        raise ValueError(f"target class must be in [0, {len(categories) - 1}]")
    model = constructor(weights=weights).eval().to(device)
    model_input = weights.transforms()(image).unsqueeze(0).to(device)
    with torch.inference_mode():
        probabilities = model(model_input)[0].softmax(0)
    order = probabilities.argsort(descending=True)
    top1_class = int(order[0])
    target_rank = int((order == target_class).nonzero(as_tuple=True)[0].item()) + 1
    result = {
        "model": name,
        "role": "black_box",
        "target_class": target_class,
        "target_label": categories[target_class],
        "target_probability": float(probabilities[target_class]),
        "target_rank": target_rank,
        "targeted_success": top1_class == target_class,
        "top1_class": top1_class,
        "top1_label": categories[top1_class],
        "top1_probability": float(probabilities[top1_class]),
    }
    del model, model_input, probabilities, order
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    if not image_path.is_file():
        raise FileNotFoundError(f"image not found: {image_path}")
    image = Image.open(image_path).convert("RGB")
    device = resolve_device(args.device)
    results = [
        evaluate_model(name, image, args.target_class, device)
        for name in args.models
    ]
    payload = {
        "image": str(image_path),
        "attack_source_model": "resnet50",
        "evaluation_protocol": "fixed_image_black_box_inference",
        "models": results,
    }
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
