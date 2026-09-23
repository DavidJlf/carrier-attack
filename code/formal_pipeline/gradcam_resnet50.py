#!/usr/bin/env python3
"""Generate a target-class Grad-CAM for the automation's ResNet-50 classifier."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50


def colorize(cam: np.ndarray) -> np.ndarray:
    """Small dependency-free blue/cyan/yellow/red heat-map."""
    stops = np.array(
        [[0, 0, 80], [0, 120, 255], [0, 255, 180], [255, 230, 0], [220, 0, 0]],
        dtype=np.float32,
    )
    position = np.clip(cam, 0.0, 1.0) * (len(stops) - 1)
    left = np.floor(position).astype(np.int32)
    right = np.minimum(left + 1, len(stops) - 1)
    alpha = (position - left)[..., None]
    return ((1.0 - alpha) * stops[left] + alpha * stops[right]).astype(np.uint8)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument(
        "--target-class",
        type=int,
        help="Class to explain. If omitted, explain the classifier's predicted Top-1 class.",
    )
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--subject-mask")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    weights = ResNet50_Weights.DEFAULT
    categories = weights.meta["categories"]
    if args.target_class is not None and not 0 <= args.target_class < len(categories):
        raise ValueError(f"target class must be in [0, {len(categories) - 1}]")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = resnet50(weights=weights).eval().to(device)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []

    def forward_hook(_module, _inputs, value):
        activations.append(value)

    def backward_hook(_module, _grad_input, grad_output):
        gradients.append(grad_output[0])

    layer = model.layer4[-1]
    forward_handle = layer.register_forward_hook(forward_hook)
    backward_handle = layer.register_full_backward_hook(backward_hook)

    image = Image.open(args.image).convert("RGB")
    tensor = weights.transforms()(image).unsqueeze(0).to(device)
    logits = model(tensor)
    probabilities = logits.softmax(1)[0]
    order = probabilities.argsort(descending=True)
    explained_class = int(order[0]) if args.target_class is None else args.target_class
    class_source = "predicted_top1" if args.target_class is None else "explicit_class"
    model.zero_grad(set_to_none=True)
    logits[0, explained_class].backward()
    forward_handle.remove()
    backward_handle.remove()

    feature = activations[0]
    gradient = gradients[0]
    channel_weights = gradient.mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((channel_weights * feature).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(224, 224), mode="bilinear", align_corners=False)[0, 0]
    cam -= cam.min()
    cam /= cam.max().clamp_min(1e-12)
    cam_np = cam.detach().cpu().numpy()

    # Save the exact spatial view classified by the model (resize 232, center crop 224).
    classifier_rgb = np.asarray(
        image.resize((232, 232), Image.Resampling.BILINEAR).crop((4, 4, 228, 228)),
        dtype=np.uint8,
    )
    heatmap = colorize(cam_np)
    overlay = (0.55 * classifier_rgb + 0.45 * heatmap).clip(0, 255).astype(np.uint8)
    Image.fromarray(classifier_rgb).save(output / "classifier_input.png")
    Image.fromarray(heatmap).save(output / "gradcam_heatmap.png")
    Image.fromarray(overlay).save(output / "gradcam_overlay.png")

    rank = int((order == explained_class).nonzero(as_tuple=True)[0].item()) + 1
    result = {
        "image": str(Path(args.image).resolve()),
        "method": "Grad-CAM",
        "model": "torchvision ResNet-50 IMAGENET1K_V2",
        "target_layer": "layer4[-1]",
        "explained_class_source": class_source,
        "explained_class": explained_class,
        "explained_label": categories[explained_class],
        "explained_probability": float(probabilities[explained_class]),
        "explained_rank": rank,
        "top1_class": int(order[0]),
        "top1_label": categories[int(order[0])],
        "top1_probability": float(probabilities[order[0]]),
    }
    if args.subject_mask:
        mask_image = Image.open(args.subject_mask).convert("L")
        mask = np.asarray(
            mask_image.resize((232, 232), Image.Resampling.NEAREST).crop((4, 4, 228, 228)),
            dtype=np.float32,
        ) / 255.0
        foreground = mask >= 0.5
        background = ~foreground
        total = float(cam_np.sum())
        result["cam_energy_foreground_fraction"] = (
            float(cam_np[foreground].sum() / total) if total > 0 and foreground.any() else None
        )
        result["cam_energy_background_fraction"] = (
            float(cam_np[background].sum() / total) if total > 0 and background.any() else None
        )
        result["subject_mask_area_fraction"] = float(foreground.mean())

    (output / "gradcam.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
