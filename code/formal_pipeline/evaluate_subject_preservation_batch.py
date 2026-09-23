#!/usr/bin/env python3
"""SAM3 and DINOv3 preservation evaluation for an automation run."""

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


METHOD_NAMES = {
    "cra": "CRA",
    "jia": "JIA",
    "cira": "CIRA",
}
CARRIER_NAMES = {
    "baseline": "Target Carrier",
    "qwen_anchor": "Non-Target Carrier",
    "hybrid_carrier": "Hybrid Carrier",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, help="Completed flux_auomation.py run directory")
    parser.add_argument("--case-index", type=Path)
    parser.add_argument("--method-results", type=Path)
    parser.add_argument("--transfer-results", type=Path)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit-cases", type=int, default=0)
    parser.add_argument("--dino-batch-pairs", type=int, default=16)
    parser.add_argument("--padding", type=float, default=0.10)
    parser.add_argument("--confidence-threshold", type=float, default=0.5)
    parser.add_argument(
        "--sam3-python", default=os.environ.get("SAM_PYTHON", sys.executable)
    )
    parser.add_argument(
        "--sam3-checkpoint",
        default=os.environ.get("SAM3_CHECKPOINT", "sam3.pt"),
    )
    parser.add_argument("--sam3-worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--manifest", type=Path, help=argparse.SUPPRESS)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_atomic(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    temp.replace(path)


def method_paths(row: dict) -> tuple[Path, Path]:
    output = Path(row["method_output"])
    clean_name = "clean.png" if row["method"] == "jia" else "input.png"
    return output / clean_name, output / "attacked.png"


def build_manifest(args: argparse.Namespace) -> dict:
    cases = read_csv(args.case_index)
    if args.limit_cases:
        cases = cases[: args.limit_cases]
    selected = {row["case_name"]: row for row in cases}
    methods = [
        row for row in read_csv(args.method_results) if row["case_name"] in selected
    ]
    images = []
    for row in methods:
        case_root = Path(row["summary_path"]).parent
        sam_meta = json.loads((case_root / "assets/sam3_meta.json").read_text())
        clean, attacked = method_paths(row)
        for role, image in (("clean", clean), ("attacked", attacked)):
            mask_dir = args.output / "masks" / row["case_name"] / row["method"]
            images.append(
                {
                    "id": f"{row['case_name']}|{row['method']}|{role}",
                    "case_name": row["case_name"],
                    "method": row["method"],
                    "role": role,
                    "image": str(image),
                    "prompt": sam_meta["prompt"],
                    "mask": str(mask_dir / f"{role}_mask.png"),
                    "meta": str(mask_dir / f"{role}_sam3.json"),
                }
            )
    return {"cases": cases, "methods": methods, "images": images}


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


def transfer_map(path: Path | None, selected_cases: set[str]) -> tuple[dict, list[str]]:
    if path is None or not path.is_file():
        return {}, []
    rows = [row for row in read_csv(path) if row["case_name"] in selected_cases]
    models = sorted({row["model"] for row in rows})
    mapping = {
        (row["case_name"], row["method"], row["model"]): row for row in rows
    }
    return mapping, models


def percentile(values, q: float):
    import numpy as np

    return float(np.percentile(values, q)) if values else ""


def summarize(rows: list[dict], black_models: list[str]) -> list[dict]:
    import numpy as np

    output = []
    for method in METHOD_NAMES:
        for variant in CARRIER_NAMES:
            group = [
                row for row in rows if row["method"] == method and row["variant"] == variant
            ]
            if not group:
                continue
            valid_dino = [float(row["dino_cls_cosine"]) for row in group if row["dino_cls_cosine"] != ""]
            valid_iou = [float(row["mask_iou"]) for row in group if row["mask_iou"] != ""]
            successful = [row for row in group if int(row["white_top1_success"]) == 1]
            success_dino = [float(row["dino_cls_cosine"]) for row in successful if row["dino_cls_cosine"] != ""]
            success_iou = [float(row["mask_iou"]) for row in successful if row["mask_iou"] != ""]
            record = {
                "method": METHOD_NAMES[method],
                "carrier": CARRIER_NAMES[variant],
                "n": len(group),
                "white_top1_asr_percent": 100 * sum(int(row["white_top1_success"]) for row in group) / len(group),
                "white_top5_asr_percent": 100 * sum(int(row["white_top5_success"]) for row in group) / len(group),
                "sam3_clean_detection_percent": 100 * sum(int(row["clean_detected"]) for row in group) / len(group),
                "sam3_adv_detection_percent": 100 * sum(int(row["adv_detected"]) for row in group) / len(group),
                "dino_mean": float(np.mean(valid_dino)) if valid_dino else "",
                "dino_median": float(np.median(valid_dino)) if valid_dino else "",
                "dino_std": float(np.std(valid_dino)) if valid_dino else "",
                "dino_p05": percentile(valid_dino, 5),
                "mask_iou_mean": float(np.mean(valid_iou)) if valid_iou else "",
                "mask_iou_median": float(np.median(valid_iou)) if valid_iou else "",
                "mask_iou_std": float(np.std(valid_iou)) if valid_iou else "",
                "mask_iou_p05": percentile(valid_iou, 5),
                "success_n": len(successful),
                "success_dino_mean": float(np.mean(success_dino)) if success_dino else "",
                "success_dino_median": float(np.median(success_dino)) if success_dino else "",
                "success_dino_p05": percentile(success_dino, 5),
                "success_mask_iou_mean": float(np.mean(success_iou)) if success_iou else "",
                "success_mask_iou_median": float(np.median(success_iou)) if success_iou else "",
                "success_mask_iou_p05": percentile(success_iou, 5),
            }
            for model_name in black_models:
                field = f"black_{model_name}_top1_success"
                record[f"black_{model_name}_top1_asr_percent"] = (
                    100 * sum(int(row[field]) for row in group) / len(group)
                )
            output.append(record)
    return output


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


def run_main(args: argparse.Namespace) -> None:
    required = [args.case_index, args.method_results, args.model, args.output]
    if any(item is None for item in required):
        raise SystemExit("--case-index, --method-results, --model and --output are required")
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = build_manifest(args)
    manifest_path = args.output / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    plan = {
        "mode": "execute" if args.execute else "dry_run",
        "cases": len(manifest["cases"]),
        "method_pairs": len(manifest["methods"]),
        "sam3_images": len(manifest["images"]),
        "variants": sorted({row["variant"] for row in manifest["cases"]}),
        "output": str(args.output),
    }
    print(json.dumps(plan, indent=2), flush=True)
    if not args.execute:
        print("DRY_RUN_OK: add --execute to launch formal evaluation.")
        return

    worker = [
        args.sam3_python,
        str(Path(__file__).resolve()),
        "--sam3-worker",
        "--manifest",
        str(manifest_path),
        "--sam3-checkpoint",
        args.sam3_checkpoint,
        "--confidence-threshold",
        str(args.confidence_threshold),
    ]
    if args.resume:
        worker.append("--resume")
    subprocess.run(worker, check=True, cwd=os.environ.get("SAM3_REPO") or None)

    install_torch_compatibility()
    import numpy as np
    import torch
    import torch.nn.functional as F
    from PIL import Image
    from transformers import AutoImageProcessor, AutoModel

    existing_path = args.output / "joined_asr_preservation_3600.csv"
    existing = read_csv(existing_path) if args.resume and existing_path.is_file() else []
    existing_map = {(row["case_name"], row["method"]): row for row in existing}
    selected_cases = {row["case_name"] for row in manifest["cases"]}
    black_map, black_models = transfer_map(args.transfer_results, selected_cases)
    processor = AutoImageProcessor.from_pretrained(args.model, local_files_only=True)
    model = AutoModel.from_pretrained(
        args.model, local_files_only=True, dtype=torch.float32
    ).eval().cuda()

    rows = list(existing)
    pending = [
        row for row in manifest["methods"] if (row["case_name"], row["method"]) not in existing_map
    ]
    started = time.time()
    for batch_start in range(0, len(pending), args.dino_batch_pairs):
        batch = pending[batch_start : batch_start + args.dino_batch_pairs]
        images = []
        records = []
        valid_positions = []
        for row in batch:
            clean_path, adv_path = method_paths(row)
            mask_dir = args.output / "masks" / row["case_name"] / row["method"]
            clean_meta = json.loads((mask_dir / "clean_sam3.json").read_text())
            adv_meta = json.loads((mask_dir / "attacked_sam3.json").read_text())
            record = dict(row)
            record["method_display"] = METHOD_NAMES[row["method"]]
            record["carrier"] = CARRIER_NAMES[row["variant"]]
            record["clean_path"] = str(clean_path)
            record["attacked_path"] = str(adv_path)
            record["clean_detected"] = int(bool(clean_meta.get("detected")))
            record["adv_detected"] = int(bool(adv_meta.get("detected")))
            record.update(
                {
                    "dino_cls_cosine": "",
                    "mask_iou": "",
                    "mask_dice": "",
                    "clean_mask_fraction_eval": "",
                    "adv_mask_fraction_eval": "",
                    "area_ratio_adv_over_clean": "",
                    "centroid_shift_normalized": "",
                }
            )
            for model_name in black_models:
                transfer = black_map[(row["case_name"], row["method"], model_name)]
                record[f"black_{model_name}_target_rank"] = transfer["black_target_rank"]
                record[f"black_{model_name}_target_probability"] = transfer["black_target_probability"]
                record[f"black_{model_name}_top1_success"] = transfer["black_top1_success"]
                record[f"black_{model_name}_top5_success"] = transfer["black_top5_success"]
            if record["clean_detected"] and record["adv_detected"]:
                clean_image = Image.open(clean_path).convert("RGB")
                adv_image = Image.open(adv_path).convert("RGB")
                if clean_image.size != adv_image.size:
                    raise ValueError(
                        f"Canvas mismatch {row['case_name']} {row['method']}: "
                        f"{clean_image.size} vs {adv_image.size}"
                    )
                clean_mask = load_mask(mask_dir / "clean_mask.png", clean_image.size)
                adv_mask = load_mask(mask_dir / "attacked_mask.png", adv_image.size)
                metrics = compute_mask_metrics(clean_mask, adv_mask)
                record.update(
                    {
                        "mask_iou": metrics["mask_iou"],
                        "mask_dice": metrics["mask_dice"],
                        "clean_mask_fraction_eval": metrics["clean_mask_fraction"],
                        "adv_mask_fraction_eval": metrics["adv_mask_fraction"],
                        "area_ratio_adv_over_clean": metrics["area_ratio_adv_over_clean"],
                        "centroid_shift_normalized": metrics["centroid_shift_normalized"],
                    }
                )
                crop_box = union_crop_box(clean_mask, adv_mask, args.padding)
                images.extend(
                    (
                        crop_subject(clean_path, clean_mask, crop_box),
                        crop_subject(adv_path, adv_mask, crop_box),
                    )
                )
                valid_positions.append(len(records))
            records.append(record)
        if images:
            inputs = processor(images=images, return_tensors="pt")
            inputs = {key: value.cuda(non_blocking=True) for key, value in inputs.items()}
            with torch.inference_mode():
                embeddings = F.normalize(model(**inputs).pooler_output.float(), dim=-1).cpu()
            for position_index, record_index in enumerate(valid_positions):
                records[record_index]["dino_cls_cosine"] = float(
                    (embeddings[position_index * 2] * embeddings[position_index * 2 + 1]).sum()
                )
        rows.extend(records)
        rows.sort(key=lambda item: (item["case_name"], item["method"]))
        write_csv_atomic(existing_path, rows)
        completed = batch_start + len(batch)
        progress = {
            "phase": "dino",
            "completed": completed,
            "total": len(pending),
            "elapsed_seconds": time.time() - started,
            "pairs_per_second": completed / max(time.time() - started, 1e-6),
        }
        print(json.dumps(progress), flush=True)

    summary_rows = summarize(rows, black_models)
    write_csv_atomic(args.output / "summary_by_method_carrier.csv", summary_rows)
    run_summary = {
        **plan,
        "status": "complete",
        "result_rows": len(rows),
        "black_models": black_models,
        "summary_by_method_carrier": summary_rows,
    }
    (args.output / "run_summary.json").write_text(
        json.dumps(run_summary, indent=2) + "\n"
    )
    print(json.dumps(run_summary, indent=2), flush=True)


if __name__ == "__main__":
    parsed = parse_args()
    if parsed.sam3_worker:
        if parsed.manifest is None:
            raise SystemExit("--manifest is required in SAM3 worker mode")
        run_sam3_worker(parsed)
    elif parsed.run_root is not None:
        run_single(parsed)
    else:
        run_main(parsed)
