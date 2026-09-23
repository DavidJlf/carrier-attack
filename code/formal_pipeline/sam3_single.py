#!/usr/bin/env python3
"""Run SAM3 text segmentation for one image and reject multiple subjects."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageOps

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument(
        "--checkpoint", default="/root/autodl-tmp/sam3/checkpoints/sam3.pt"
    )
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_path = Path(args.image).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.open(image_path).convert("RGB")

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint,
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        state = processor.set_image(image)
        output = processor.set_text_prompt(state=state, prompt=args.prompt)

    masks = output["masks"].detach().cpu()
    scores = output["scores"].detach().float().cpu()
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]
    instance_count = int(masks.shape[0])
    metadata = {
        "image": str(image_path),
        "prompt": args.prompt,
        "confidence_threshold": args.confidence_threshold,
        "num_instances": instance_count,
        "scores": [float(value) for value in scores.flatten()],
    }

    if instance_count == 0:
        metadata["error"] = "SAM3 did not find the requested subject."
        (output_dir / "sam3_error.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        raise SystemExit("ERROR: SAM3没有检测到目标主体，自动化已退出。")
    if instance_count > 1:
        metadata["error"] = "SAM3 found multiple requested subjects."
        (output_dir / "sam3_error.json").write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )
        raise SystemExit(
            f"ERROR: SAM3检测到{instance_count}个同类主体；因为存在多个主体，自动化已退出。"
        )

    mask = masks[0].bool().numpy()
    metadata.update(
        {
            "mask_pixels": int(mask.sum()),
            "image_pixels": int(mask.size),
            "mask_fraction": float(mask.mean()),
        }
    )
    mask_image = Image.fromarray(mask.astype(np.uint8) * 255)
    mask_image.save(output_dir / "subject_mask.png")
    ImageOps.invert(mask_image).save(output_dir / "background_mask.png")

    array = np.asarray(image, dtype=np.float32).copy()
    array[mask] = 0.55 * array[mask] + 0.45 * np.array([60, 220, 90])
    Image.fromarray(np.clip(array, 0, 255).astype(np.uint8)).save(
        output_dir / "subject_mask_overlay.png"
    )
    (output_dir / "sam3_meta.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )
    print(json.dumps(metadata, indent=2), flush=True)


if __name__ == "__main__":
    main()
