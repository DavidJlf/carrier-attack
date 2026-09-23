#!/usr/bin/env python3
"""Build a deterministic final index and metrics for the formal 1200-case run."""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path


EXPERIMENTS = Path("/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments")
ROOTS = [
    EXPERIMENTS / "formal_20x30",
    EXPERIMENTS / "formal_20x30_repair",
    EXPERIMENTS / "formal_20x30_rc_car_repair",
]
OUTPUT = EXPERIMENTS / "formal_20x30_final_summary"
METHODS = ["cra", "jia", "cira"]
MODELS = ["resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b"]


def pct(value: int, total: int) -> float:
    return round(100.0 * value / total, 6) if total else 0.0


def mean(value: float, total: int) -> float:
    return round(value / total, 8) if total else 0.0


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def aggregate(rows: list[dict], keys: tuple[str, ...], prefix: str) -> list[dict]:
    buckets = defaultdict(lambda: {"n": 0, "top1": 0, "top5": 0, "prob": 0.0})
    for row in rows:
        key = tuple(row[name] for name in keys)
        item = buckets[key]
        item["n"] += 1
        item["top1"] += int(row[f"{prefix}_top1_success"])
        item["top5"] += int(row[f"{prefix}_top5_success"])
        item["prob"] += float(row[f"{prefix}_target_probability"])
    result = []
    for key in sorted(buckets, key=lambda value: tuple(str(part) for part in value)):
        item = buckets[key]
        record = {name: value for name, value in zip(keys, key)}
        record.update({
            "n": item["n"],
            "top1_asr_percent": pct(item["top1"], item["n"]),
            "top5_asr_percent": pct(item["top5"], item["n"]),
            "mean_target_probability": mean(item["prob"], item["n"]),
        })
        result.append(record)
    return result


def main() -> None:
    expected = {p.parent.name for p in ROOTS[0].rglob("manifest.json")}
    selected: dict[str, Path] = {}
    raw_counts = {}
    for root in ROOTS:
        paths = list(root.rglob("summary.json"))
        raw_counts[root.name] = len(paths)
        for path in paths:
            name = path.parent.name
            if name in selected:
                raise RuntimeError(f"Duplicate successful case: {name}")
            selected[name] = path

    actual = set(selected)
    if expected != actual:
        raise RuntimeError(
            f"Final set mismatch: missing={sorted(expected-actual)}, unexpected={sorted(actual-expected)}"
        )

    method_rows = []
    transfer_rows = []
    case_index = []
    for case_name in sorted(selected):
        path = selected[case_name]
        summary = json.loads(path.read_text(encoding="utf-8"))
        manifest_path = path.parent / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        variant = "qwen_anchor" if case_name.endswith("_qwen_anchor") else "baseline"
        source = manifest.get("source_case") or manifest.get("source") or case_name.split("_to_", 1)[0]
        case_index.append({
            "case_name": case_name,
            "source": source,
            "target_label": summary["target_label"],
            "target_class": summary["target_class"],
            "variant": variant,
            "summary_path": str(path),
        })
        methods = summary.get("methods", [])
        if {item.get("method") for item in methods} != set(METHODS):
            raise RuntimeError(f"Incomplete methods in {path}")
        for item in methods:
            final = item.get("final_eval") or {}
            rank = int(final.get("target_rank", 1001))
            cam = item.get("gradcam") or {}
            row = {
                "case_name": case_name,
                "source": source,
                "target_label": summary["target_label"],
                "target_class": summary["target_class"],
                "variant": variant,
                "method": item["method"],
                "white_target_rank": rank,
                "white_target_probability": float(final.get("target_probability", 0.0)),
                "white_top1_success": int(rank == 1),
                "white_top5_success": int(rank <= 5),
                "gradcam_foreground_fraction": cam.get("cam_energy_foreground_fraction"),
                "gradcam_background_fraction": cam.get("cam_energy_background_fraction"),
                "subject_mask_area_fraction": cam.get("subject_mask_area_fraction"),
                "summary_path": str(path),
                "method_output": item.get("output"),
            }
            method_rows.append(row)
            models = (item.get("transfer_eval") or {}).get("models", [])
            if {model.get("model") for model in models} != set(MODELS):
                raise RuntimeError(f"Incomplete transfer models in {path}: {item['method']}")
            for model in models:
                black_rank = int(model.get("target_rank", 1001))
                transfer_rows.append({
                    "case_name": case_name,
                    "source": source,
                    "target_label": summary["target_label"],
                    "target_class": summary["target_class"],
                    "variant": variant,
                    "method": item["method"],
                    "model": model["model"],
                    "black_target_rank": black_rank,
                    "black_target_probability": float(model.get("target_probability", 0.0)),
                    "black_top1_success": int(black_rank == 1),
                    "black_top5_success": int(black_rank <= 5),
                })

    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "case_index_1200.csv", case_index)
    write_csv(OUTPUT / "method_results_3600.csv", method_rows)
    write_csv(OUTPUT / "transfer_results_18000.csv", transfer_rows)
    white_method_variant = aggregate(method_rows, ("method", "variant"), "white")
    white_method = aggregate(method_rows, ("method",), "white")
    white_source_method_variant = aggregate(method_rows, ("source", "method", "variant"), "white")
    black_method_model_variant = aggregate(transfer_rows, ("method", "model", "variant"), "black")
    black_method_model = aggregate(transfer_rows, ("method", "model"), "black")
    black_source_method_model_variant = aggregate(
        transfer_rows, ("source", "method", "model", "variant"), "black"
    )
    write_csv(OUTPUT / "white_by_method.csv", white_method)
    write_csv(OUTPUT / "white_by_method_variant.csv", white_method_variant)
    write_csv(OUTPUT / "white_by_source_method_variant.csv", white_source_method_variant)
    write_csv(OUTPUT / "black_by_method_model.csv", black_method_model)
    write_csv(OUTPUT / "black_by_method_model_variant.csv", black_method_model_variant)
    write_csv(OUTPUT / "black_by_source_method_model_variant.csv", black_source_method_model_variant)
    report = {
        "status": "complete",
        "expected_cases": len(expected),
        "unique_cases": len(case_index),
        "method_results": len(method_rows),
        "transfer_results": len(transfer_rows),
        "raw_summary_counts": raw_counts,
        "missing_cases": [],
        "duplicate_cases": [],
        "methods": METHODS,
        "black_box_models": MODELS,
        "white_by_method": white_method,
        "white_by_method_variant": white_method_variant,
        "black_by_method_model": black_method_model,
        "black_by_method_model_variant": black_method_model_variant,
    }
    (OUTPUT / "final_summary.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
