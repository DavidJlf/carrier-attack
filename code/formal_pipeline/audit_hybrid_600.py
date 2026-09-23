#!/usr/bin/env python3
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments/formal_20x30_hybrid_carrier")
OUT = Path("/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments/formal_20x30_hybrid_carrier_audit")
METHODS = {"cra", "jia", "cira"}
MODELS = {"resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b"}

summaries = sorted(ROOT.glob("worker_[0-9][0-9]/*/summary.json"))
case_locations = defaultdict(list)
source_counts = Counter()
target_counts = Counter()
method_rows = 0
transfer_rows = 0
problems = []
case_rows = []
method_records = []
transfer_records = []

for path in summaries:
    case_locations[path.parent.name].append(str(path))
    try:
        summary = json.loads(path.read_text())
        manifest = json.loads((path.parent / "manifest.json").read_text())
    except Exception as exc:
        problems.append({"case": path.parent.name, "error": f"json:{exc}"})
        continue
    source_counts[manifest.get("source_case") or manifest.get("source") or manifest.get("subject")] += 1
    source = manifest.get("source_case") or manifest.get("source") or manifest.get("subject")
    target_counts[str(summary.get("target_class"))] += 1
    case_rows.append({"case_name": path.parent.name, "source": source, "target_label": summary.get("target_label"), "target_class": summary.get("target_class"), "variant": "hybrid_carrier", "summary_path": str(path)})
    methods = summary.get("methods", [])
    names = {item.get("method") for item in methods}
    if names != METHODS or len(methods) != 3:
        problems.append({"case": path.parent.name, "error": "methods", "actual": sorted(str(x) for x in names)})
    for item in methods:
        method_rows += 1
        method = item.get("method")
        output = Path(item.get("output", ""))
        clean = output / ("clean.png" if method == "jia" else "input.png")
        attacked = output / "attacked.png"
        if not clean.is_file() or not attacked.is_file():
            problems.append({"case": path.parent.name, "method": method, "error": "missing_clean_or_attacked"})
        models = (item.get("transfer_eval") or {}).get("models", [])
        model_names = {model.get("model") for model in models}
        transfer_rows += len(models)
        if model_names != MODELS or len(models) != 5:
            problems.append({"case": path.parent.name, "method": method, "error": "transfer_models", "actual": sorted(str(x) for x in model_names)})
        final = item.get("final_eval") or {}
        rank = int(final.get("target_rank", 1001))
        method_records.append({"case_name": path.parent.name, "source": source, "target_label": summary.get("target_label"), "target_class": summary.get("target_class"), "variant": "hybrid_carrier", "method": method, "white_target_rank": rank, "white_target_probability": float(final.get("target_probability", 0.0)), "white_top1_success": int(rank == 1), "white_top5_success": int(rank <= 5), "summary_path": str(path), "method_output": str(output)})
        for model in models:
            black_rank = int(model.get("target_rank", 1001))
            transfer_records.append({"case_name": path.parent.name, "source": source, "target_label": summary.get("target_label"), "target_class": summary.get("target_class"), "variant": "hybrid_carrier", "method": method, "model": model.get("model"), "black_target_rank": black_rank, "black_target_probability": float(model.get("target_probability", 0.0)), "black_top1_success": int(black_rank == 1), "black_top5_success": int(black_rank <= 5)})

duplicates = {name: paths for name, paths in case_locations.items() if len(paths) != 1}
failure_markers = [str(p) for p in ROOT.glob("worker_[0-9][0-9]/*/.*_failed_skip")]
report = {
    "status": "complete" if len(summaries) == 600 and len(case_locations) == 600 and method_rows == 1800 and transfer_rows == 9000 and not duplicates and not failure_markers and not problems else "failed",
    "summary_files": len(summaries),
    "unique_cases": len(case_locations),
    "method_rows": method_rows,
    "transfer_rows": transfer_rows,
    "source_count": len(source_counts),
    "target_count": len(target_counts),
    "source_distribution": dict(sorted(source_counts.items())),
    "target_distribution": dict(sorted(target_counts.items())),
    "duplicates": duplicates,
    "failure_markers": failure_markers,
    "problems": problems,
}
OUT.mkdir(parents=True, exist_ok=True)
(OUT / "audit.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
def write_csv(name, rows):
    with (OUT / name).open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(sorted(rows, key=lambda row: tuple(str(row.get(key, "")) for key in ("case_name", "method", "model"))))
if report["status"] == "complete":
    write_csv("case_index_600.csv", case_rows)
    write_csv("method_results_1800.csv", method_records)
    write_csv("transfer_results_9000.csv", transfer_records)
print(json.dumps(report, ensure_ascii=False))
raise SystemExit(0 if report["status"] == "complete" else 1)
