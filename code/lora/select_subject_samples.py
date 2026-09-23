from __future__ import annotations

import argparse
import csv
import collections
import math
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

from sam3.model.sam3_image_processor import Sam3Processor
from sam3.model_builder import build_sam3_image_model


def sharpness(gray: np.ndarray, mask: np.ndarray) -> float:
    gx = np.diff(gray, axis=1, prepend=gray[:, :1])
    gy = np.diff(gray, axis=0, prepend=gray[:1, :])
    values = gx * gx + gy * gy
    return float(values[mask].mean()) if mask.any() else 0.0


def make_contacts(paths: list[Path], output: Path, prefix: str, per_sheet: int = 40) -> None:
    output.mkdir(parents=True, exist_ok=True)
    thumb = 256
    cols, rows = 5, 8
    for sheet_i in range(math.ceil(len(paths) / per_sheet)):
        subset = paths[sheet_i * per_sheet:(sheet_i + 1) * per_sheet]
        canvas = Image.new("RGB", (cols * thumb, rows * (thumb + 28)), "white")
        draw = ImageDraw.Draw(canvas)
        for j, path in enumerate(subset):
            im = Image.open(path).convert("RGB")
            im.thumbnail((thumb, thumb))
            x = (j % cols) * thumb + (thumb - im.width) // 2
            y = (j // cols) * (thumb + 28)
            canvas.paste(im, (x, y))
            draw.text(((j % cols) * thumb + 4, y + thumb + 4), path.stem, fill="black")
        canvas.save(output / f"{prefix}_{sheet_i + 1:02d}.jpg", quality=92)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--prompt", choices=["cat", "dog"], required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--keep", type=int, default=40)
    p.add_argument("--metadata")
    args = p.parse_args()

    image_paths = sorted(Path(args.input).glob("*.png"))
    if len(image_paths) < args.keep:
        raise RuntimeError(f"Need at least {args.keep} PNG files, found {len(image_paths)}")
    out = Path(args.output)
    selected_dir = out / "selected_top40"
    selected_dir.mkdir(parents=True, exist_ok=True)

    model = build_sam3_image_model(
        checkpoint_path=args.checkpoint, load_from_HF=False, device="cuda",
        eval_mode=True, enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)
    prompt_by_name = {}
    if args.metadata:
        with Path(args.metadata).open("r", newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                prompt_by_name[Path(row["image"]).name] = row.get("prompt", "")
    records = []
    with torch.inference_mode(), torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for i, path in enumerate(image_paths, 1):
            image = Image.open(path).convert("RGB")
            state = processor.set_image(image)
            result = processor.set_text_prompt(state=state, prompt=args.prompt)
            masks = result["masks"].detach().cpu()
            scores = result["scores"].detach().float().cpu()
            if masks.ndim == 4 and masks.shape[1] == 1:
                masks = masks[:, 0]
            if masks.shape[0]:
                union = masks.bool().any(dim=0).numpy()
                sam_score = float(scores.max()) if scores.numel() else 0.0
            else:
                union = np.zeros((image.height, image.width), dtype=bool)
                sam_score = 0.0
            rgb = np.asarray(image, dtype=np.float32) / 255.0
            gray = rgb.mean(axis=2)
            area = float(union.mean())
            subject_sharpness = sharpness(gray, union)
            exposure = float(gray[union].mean()) if union.any() else 0.0
            exposure_penalty = max(0.0, 1.0 - abs(exposure - 0.5) * 1.4)
            area_penalty = 1.0 if 0.04 <= area <= 0.75 else 0.5
            quality = sam_score * math.log1p(subject_sharpness * 10000.0) * exposure_penalty * area_penalty
            ys, xs = np.where(union)
            if len(xs):
                cx, cy = float(xs.mean() / union.shape[1]), float(ys.mean() / union.shape[0])
                bw = float((xs.max() - xs.min() + 1) / union.shape[1])
                bh = float((ys.max() - ys.min() + 1) / union.shape[0])
            else:
                cx = cy = bw = bh = 0.0
            prompt = prompt_by_name.get(path.name, "")
            records.append({
                "image": str(path), "sam_score": sam_score, "mask_fraction": area,
                "subject_sharpness": subject_sharpness, "subject_exposure": exposure,
                "quality_score": quality, "prompt": prompt, "cx": cx, "cy": cy,
                "bbox_w": bw, "bbox_h": bh,
            })
            print(f"[{i:03d}/200] score={quality:.4f} sam={sam_score:.4f} {path.name}", flush=True)

    records.sort(key=lambda x: x["quality_score"], reverse=True)
    shots = ["close-up portrait", "full-body photograph", "side-profile photograph", "low-angle photograph", "eye-level photograph"]
    activities = ["sitting calmly", "walking naturally", "looking toward the camera", "resting comfortably", "exploring the surroundings"]
    scenes = ["window", "garden", "chair", "beach", "snowy", "city street", "blanket", "forest", "studio", "flowers", "cardboard box", "lake", "stone steps", "living room", "wooden table", "grassy field", "windowsill", "fountain", "country road", "library"]
    def tag(text, choices):
        return next((x for x in choices if x in text), "unknown")
    selected = []
    counts = {"shot": collections.Counter(), "activity": collections.Counter(), "scene": collections.Counter(), "pose": collections.Counter()}
    # First pass: strict quotas. Pose bins come from the SAM mask geometry.
    for row in records:
        shot = tag(row["prompt"], shots); activity = tag(row["prompt"], activities); scene = tag(row["prompt"], scenes)
        pose = (round(row["cx"] * 3), round(row["cy"] * 3), round(row["bbox_w"] * 3), round(row["bbox_h"] * 3))
        if counts["shot"][shot] >= 10 or counts["activity"][activity] >= 10 or counts["scene"][scene] >= 4 or counts["pose"][pose] >= 5:
            continue
        selected.append(row); counts["shot"][shot] += 1; counts["activity"][activity] += 1; counts["scene"][scene] += 1; counts["pose"][pose] += 1
        if len(selected) == args.keep: break
    # Fill any shortfall by quality while avoiding exact duplicates.
    for row in records:
        if len(selected) == args.keep: break
        if row not in selected: selected.append(row)
    selected_ids = {id(row): rank for rank, row in enumerate(selected, 1)}
    for quality_rank, row in enumerate(records, 1):
        row["quality_rank"] = quality_rank
        row["selected_rank"] = selected_ids.get(id(row), "")
    for row in selected:
        shutil.copy2(row["image"], selected_dir / Path(row["image"]).name)
    fields = ["quality_rank", "selected_rank", "image", "quality_score", "sam_score", "mask_fraction", "subject_sharpness", "subject_exposure", "cx", "cy", "bbox_w", "bbox_h", "prompt"]
    with (out / "quality_ranking.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader(); w.writerows(records)
    make_contacts(image_paths, out / "contact_remaining", "remaining")
    make_contacts([Path(r["image"]) for r in selected], out / "contact_selected", "selected_top40")


if __name__ == "__main__":
    main()
