#!/usr/bin/env python3
"""Evaluate one image with the official torchvision ResNet-50 preprocessing."""

import argparse
import json
from pathlib import Path

import torch
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--target-class", required=True, type=int)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    weights = ResNet50_Weights.DEFAULT
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=weights).eval().to(device)
    image = Image.open(args.image).convert("RGB")
    with torch.no_grad():
        logits = model(weights.transforms()(image).unsqueeze(0).to(device))[0]
        probabilities = logits.softmax(0)
    order = probabilities.argsort(descending=True)
    rank = int((order == args.target_class).nonzero(as_tuple=True)[0].item()) + 1
    result = {
        "target_class": args.target_class,
        "target_label": weights.meta["categories"][args.target_class],
        "target_probability": float(probabilities[args.target_class]),
        "target_rank": rank,
        "top1_class": int(order[0]),
        "top1_label": weights.meta["categories"][int(order[0])],
        "top1_probability": float(probabilities[order[0]]),
        "top5": [
            {
                "class": int(index),
                "label": weights.meta["categories"][int(index)],
                "probability": float(probabilities[index]),
            }
            for index in order[:5]
        ],
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
