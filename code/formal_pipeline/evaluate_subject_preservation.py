#!/usr/bin/env python3
"""SAM3 and DINOv3 subject-preservation evaluation for one automation run."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument("--sam3-python", default=os.environ.get("SAM_PYTHON", sys.executable))
    parser.add_argument("--sam3-checkpoint", default=os.environ.get("SAM3_CHECKPOINT", "sam3.pt"))
    parser.add_argument("--sam3-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def prepare_sam3_import() -> None:
    script_parent = str(Path(__file__).resolve().parent)
    sys.path = [
        entry
        for entry in sys.path
        if str(Path(entry or ".").resolve()) != script_parent
    ]
    sam3_repo = os.environ.get("SAM3_REPO")
    if sam3_repo:
        sys.path.insert(0, sam3_repo)


def run_sam3_worker(args: argparse.Namespace) -> None:
    prepare_sam3_import()
    import numpy as np
    import torch
    from PIL import Image
    from sam3.model.sam3_image_processor import Sam3Processor
    from sam3.model_builder import build_sam3_image_model

    manifest = json.loads(args.manifest.read_text())
    entries = manifest["images"]
    model = build_sam3_image_model(
        checkpoint_path=args.sam3_checkpoint,
        load_from_HF=False,
        device="cuda",
        eval_mode=True,
        enable_inst_interactivity=False,
    )
    processor = Sam3Processor(model, confidence_threshold=args.confidence_threshold)
    progress_path = args.manifest.parent / "sam3_progress.jsonl"
    started = time.time()
    completed = 0
    for index, item in enumerate(entries, 1):
        mask_path = Path(item["mask"])
        meta_path = Path(item["meta"])
        if args.resume and mask_path.is_file() and meta_path.is_file():
            try:
                meta = json.loads(meta_path.read_text())
                if meta.get("detected") and meta.get("image") == item["image"]:
                    completed += 1
                    continue
            except Exception:
                pass
        image_path = Path(item["image"])
        image = Image.open(image_path).convert("RGB")
        with torch.inference_mode(), torch.autocast(
            device_type="cuda", dtype=torch.bfloat16
        ):
            state = processor.set_image(image)
            output = processor.set_text_prompt(state=state, prompt=item["prompt"])
        masks = output["masks"].detach().cpu()
        scores = output["scores"].detach().float().cpu()
        if masks.ndim == 4 and masks.shape[1] == 1:
            masks = masks[:, 0]
        count = int(masks.shape[0])
        meta = {
            "id": item["id"],
            "image": item["image"],
            "prompt": item["prompt"],
            "confidence_threshold": args.confidence_threshold,
            "num_instances": count,
            "scores": [float(value) for value in scores.flatten()],
            "mask_mode": "union_all_instances",
            "detected": count > 0,
        }
        mask_path.parent.mkdir(parents=True, exist_ok=True)
        if count:
            mask = masks.bool().any(dim=0).numpy()
            Image.fromarray(mask.astype(np.uint8) * 255).save(mask_path)
            meta.update(
                {
                    "merged_instance_count": count,
                    "mask_pixels": int(mask.sum()),
                    "image_pixels": int(mask.size),
                    "mask_fraction": float(mask.mean()),
                }
            )
        meta_path.write_text(json.dumps(meta, indent=2) + "\n")
        completed += 1
        if index % 25 == 0 or index == len(entries):
            record = {
                "phase": "sam3",
                "completed": completed,
                "total": len(entries),
                "elapsed_seconds": time.time() - started,
                "images_per_second": completed / max(time.time() - started, 1e-6),
            }
            with progress_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)


def install_torch_compatibility() -> None:
    import torch
    import torch.utils._pytree as torch_pytree

    if not hasattr(torch_pytree, "register_pytree_node"):
        def compat(typ, flatten_fn, unflatten_fn, **kwargs):
            kwargs.pop("serialized_type_name", None)
            return torch_pytree._register_pytree_node(
                typ, flatten_fn, unflatten_fn, **kwargs
            )
        torch_pytree.register_pytree_node = compat
    if hasattr(torch, "compiler") and not hasattr(torch.compiler, "is_compiling"):
        torch.compiler.is_compiling = lambda: False


def load_mask(path: Path, size: tuple[int, int]):
    import numpy as np
    from PIL import Image

    mask = Image.open(path).convert("L")
    if mask.size != size:
        mask = mask.resize(size, Image.Resampling.NEAREST)
    return np.asarray(mask) >= 128


def bbox(mask):
    import numpy as np

    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max() + 1), int(ys.max() + 1)


def union_crop_box(clean_mask, adv_mask, padding: float):
    boxes = [item for item in (bbox(clean_mask), bbox(adv_mask)) if item]
    if not boxes:
        return None
    left = min(item[0] for item in boxes)
    top = min(item[1] for item in boxes)
    right = max(item[2] for item in boxes)
    bottom = max(item[3] for item in boxes)
    pad_x = round((right - left) * padding)
    pad_y = round((bottom - top) * padding)
    height, width = clean_mask.shape
    return (
        max(0, left - pad_x),
        max(0, top - pad_y),
        min(width, right + pad_x),
        min(height, bottom + pad_y),
    )


def crop_subject(image_path: Path, mask, crop_box):
    import numpy as np
    from PIL import Image

    image = Image.open(image_path).convert("RGB")
    array = np.asarray(image).copy()
    array[~mask] = 128
    return Image.fromarray(array).crop(crop_box)


def compute_mask_metrics(clean_mask, adv_mask) -> dict:
    import numpy as np

    intersection = int(np.logical_and(clean_mask, adv_mask).sum())
    union = int(np.logical_or(clean_mask, adv_mask).sum())
    clean_area = int(clean_mask.sum())
    adv_area = int(adv_mask.sum())
    clean_y, clean_x = np.nonzero(clean_mask)
    adv_y, adv_x = np.nonzero(adv_mask)
    diagonal = math.hypot(*clean_mask.shape)
    centroid_shift = None
    if len(clean_x) and len(adv_x):
        centroid_shift = math.hypot(
            float(clean_x.mean() - adv_x.mean()),
            float(clean_y.mean() - adv_y.mean()),
        ) / diagonal
    return {
        "mask_iou": intersection / union if union else 0.0,
        "mask_dice": 2 * intersection / (clean_area + adv_area)
        if clean_area + adv_area
        else 0.0,
        "clean_mask_fraction": clean_area / clean_mask.size,
        "adv_mask_fraction": adv_area / adv_mask.size,
        "area_ratio_adv_over_clean": adv_area / clean_area if clean_area else "",
        "centroid_shift_normalized": centroid_shift if centroid_shift is not None else "",
    }


def run_single(args: argparse.Namespace) -> None:
    """Apply the existing independent SAM3/DINOv3 measurements to one run."""
    if args.model is None:
        raise SystemExit("--model is required with --run-root")
    run_root = args.run_root.expanduser().resolve()
    summary = json.loads((run_root / "summary.json").read_text(encoding="utf-8"))
    sam_meta = json.loads((run_root / "assets" / "sam3_meta.json").read_text(encoding="utf-8"))
    output_dir = args.output or run_root / "subject_preservation"
    output_dir.mkdir(parents=True, exist_ok=True)
    pairs = []
    entries = []
    for stage in summary["methods"]:
        method = stage["method"]
        method_dir = Path(stage["output"])
        clean = method_dir / ("clean.png" if method == "jia" else "input.png")
        attacked = method_dir / "attacked.png"
        if not clean.is_file() or not attacked.is_file():
            raise FileNotFoundError(f"Missing clean/attacked pair for {method}: {clean}, {attacked}")
        pairs.append((method, clean, attacked))
        for role, image_path in (("clean", clean), ("attacked", attacked)):
            mask_dir = output_dir / "masks" / method
            entries.append({
                "id": f"{method}|{role}", "method": method, "role": role,
                "image": str(image_path), "prompt": sam_meta["prompt"],
                "mask": str(mask_dir / f"{role}_mask.png"),
                "meta": str(mask_dir / f"{role}_sam3.json"),
            })
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps({"images": entries}, indent=2), encoding="utf-8")
    print(json.dumps({"run_root": str(run_root), "pairs": len(pairs), "mode": "execute" if args.execute else "dry_run"}), flush=True)
    if not args.execute:
        return
    worker = [
        args.sam3_python, str(Path(__file__).resolve()), "--sam3-worker",
        "--manifest", str(manifest_path), "--sam3-checkpoint", args.sam3_checkpoint,
        "--confidence-threshold", str(args.confidence_threshold),
    ]
    if args.resume:
        worker.append("--resume")
    subprocess.run(worker, check=True, cwd=os.environ.get("SAM3_REPO") or None)

    install_torch_compatibility()
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(args.model, local_files_only=True, torch_dtype=torch.float32).eval().cuda()
    rows = []
    for method, clean, attacked in pairs:
        mask_dir = output_dir / "masks" / method
        clean_meta = json.loads((mask_dir / "clean_sam3.json").read_text(encoding="utf-8"))
        adv_meta = json.loads((mask_dir / "attacked_sam3.json").read_text(encoding="utf-8"))
        row = {"method": method, "clean_detected": bool(clean_meta["detected"]),
               "attacked_detected": bool(adv_meta["detected"]), "dino_cls_cosine": "", "mask_iou": ""}
        if row["clean_detected"] and row["attacked_detected"]:
            clean_image = Image.open(clean).convert("RGB")
            attacked_image = Image.open(attacked).convert("RGB")
            if clean_image.size != attacked_image.size:
                raise ValueError(f"Canvas mismatch for {method}: {clean_image.size} vs {attacked_image.size}")
            clean_mask = load_mask(mask_dir / "clean_mask.png", clean_image.size)
            adv_mask = load_mask(mask_dir / "attacked_mask.png", attacked_image.size)
            row.update(compute_mask_metrics(clean_mask, adv_mask))
            crop_box = union_crop_box(clean_mask, adv_mask, args.padding)
            images = [crop_subject(clean, clean_mask, crop_box), crop_subject(attacked, adv_mask, crop_box)]
            inputs = {key: value.cuda() for key, value in processor(images=images, return_tensors="pt").items()}
            with torch.inference_mode():
                embeddings = F.normalize(model(**inputs).pooler_output.float(), dim=-1)
            row["dino_cls_cosine"] = float((embeddings[0] * embeddings[1]).sum().cpu())
        rows.append(row)
    write_csv_atomic(output_dir / "preservation_by_method.csv", rows)
    (output_dir / "preservation.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(json.dumps(rows, indent=2), flush=True)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.sam3_worker:
        if parsed.manifest is None:
            raise SystemExit("--manifest is required in SAM3 worker mode")
        run_sam3_worker(parsed)
    else:
        if parsed.run_root is None or parsed.model is None:
            raise SystemExit("--run-root and --model are required")
        run_single(parsed)
