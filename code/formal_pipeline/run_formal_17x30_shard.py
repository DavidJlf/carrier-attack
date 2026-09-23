#!/usr/bin/env python3
"""Run one deterministic shard of the formal 20-source x 30-target experiment."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess

from anchor_catalog import lookup_anchor
from hybrid_prompt_catalog import (
    DEFAULT_CATALOG as DEFAULT_HYBRID_CATALOG,
    build_hybrid_prompts,
    load_hybrid_catalog,
    validate_formal_mapping,
)
from formal_runner_common import (
    ANCHOR_CATALOG,
    ANCHOR_GATE,
    AUTOMATION,
    METHODS,
    PYTHON,
    automation_command,
    execute,
    slug,
)


HERE = Path(__file__).resolve().parent
SOURCE_CSV = HERE / "dreambooth_20_sources.csv"
SOURCE_GATE_CATALOG = HERE / "dreambooth_source_gate_catalog.json"
DEFAULT_OUTPUT_BASE = Path("/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments/formal_20x30")
DEFAULT_HYBRID_OUTPUT_BASE = Path(
    "/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments/formal_20x30_hybrid_carrier"
)

TARGETS = (
    {"class": 94, "label": "hummingbird", "group": "animal_small"},
    {"class": 113, "label": "snail", "group": "animal_small"},
    {"class": 301, "label": "ladybug", "group": "animal_small"},
    {"class": 331, "label": "hare", "group": "animal_small"},
    {"class": 272, "label": "coyote", "group": "animal_medium"},
    {"class": 277, "label": "red fox", "group": "animal_medium"},
    {"class": 340, "label": "zebra", "group": "animal_medium"},
    {"class": 348, "label": "ram", "group": "animal_medium"},
    {"class": 344, "label": "hippopotamus", "group": "animal_large"},
    {"class": 347, "label": "bison", "group": "animal_large"},
    {"class": 354, "label": "Arabian camel", "group": "animal_large"},
    {"class": 386, "label": "African elephant", "group": "animal_large"},
    {"class": 487, "label": "cellular telephone", "group": "small_object"},
    {"class": 504, "label": "coffee mug", "group": "small_object"},
    {"class": 626, "label": "lighter", "group": "small_object"},
    {"class": 644, "label": "matchstick", "group": "small_object"},
    {"class": 695, "label": "padlock", "group": "small_object"},
    {"class": 709, "label": "pencil box", "group": "small_object"},
    {"class": 722, "label": "ping-pong ball", "group": "small_object"},
    {"class": 761, "label": "remote control", "group": "small_object"},
    {"class": 852, "label": "tennis ball", "group": "small_object"},
    {"class": 902, "label": "whistle", "group": "small_object"},
    {"class": 404, "label": "airliner", "group": "vehicle"},
    {"class": 407, "label": "ambulance", "group": "vehicle"},
    {"class": 555, "label": "fire engine", "group": "vehicle"},
    {"class": 717, "label": "pickup", "group": "vehicle"},
    {"class": 779, "label": "school bus", "group": "vehicle"},
    {"class": 483, "label": "castle", "group": "scene"},
    {"class": 562, "label": "fountain", "group": "scene"},
    {"class": 580, "label": "greenhouse", "group": "scene"},
)

SUBJECTS = {
    "backpack": "backpack", "bear_plushie": "bear plushie", "candle": "candle",
    "cat": "cat", "cat2": "cat", "dog": "dog", "dog2": "dog",
    "dog3": "dog", "dog5": "dog", "dog7": "dog", "fancy_boot": "boot",
    "grey_sloth_plushie": "sloth plushie", "pink_sunglasses": "sunglasses",
    "rc_car": "toy car", "shiny_sneaker": "sneaker", "teapot": "teapot",
    "vase": "vase", "duck_toy": "duck toy", "poop_emoji": "poop emoji",
    "robot_toy": "robot toy",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--num-shards", type=int, default=8,
        help="Total formal workers (default: 8 for master plus cards 1-7).",
    )
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT_BASE)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--repair-case", action="append", default=[], metavar="JOB_ID:VARIANT",
        help="Run only an exact repair variant; repeat as needed.",
    )
    parser.add_argument("--seed-offset", type=int, default=0)
    parser.add_argument("--attack-scale", type=float, default=0.5)
    parser.add_argument(
        "--construction-condition", choices=("legacy", "hybrid"), default="legacy",
        help="legacy preserves completed Target/Non-Target variants; hybrid runs only the supplemental Hybrid Carrier condition.",
    )
    parser.add_argument(
        "--hybrid-prompt-catalog", type=Path, default=DEFAULT_HYBRID_CATALOG,
    )
    parser.add_argument(
        "--subject-override", action="append", default=[], metavar="SLUG=TEXT",
        help="Override the SAM3 subject prompt for selected repair runs.",
    )
    return parser.parse_args()


def seed_from_path(path: Path) -> int:
    match = re.search(r"seed(\d+)", path.name)
    if not match:
        raise ValueError(f"Cannot resolve seed from source filename: {path}")
    return int(match.group(1))


def load_sources() -> list[dict]:
    catalog = json.loads(SOURCE_GATE_CATALOG.read_text(encoding="utf-8-sig"))["cases"]
    rows = list(csv.DictReader(SOURCE_CSV.open(encoding="utf-8-sig", newline="")))
    sources = []
    for row in rows:
        case = catalog.get(row["concept_slug"])
        if not case or case.get("enabled") is not True:
            continue
        classes = [int(value) for value in case.get("classes", [])]
        source_image = Path(row["source_image"])
        sources.append({
            "slug": row["concept_slug"],
            "source_case": row["concept_slug"],
            "subject": SUBJECTS[row["concept_slug"]],
            "source_classes": ",".join(str(value) for value in classes),
            "image": source_image,
            "lora": Path(row["lora_path"]),
            "seed": seed_from_path(source_image),
        })
    return sources


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    sources = load_sources()
    subject_overrides = {}
    for item in args.subject_override:
        if "=" not in item:
            raise SystemExit(f"Invalid --subject-override {item!r}; expected SLUG=TEXT")
        override_slug, text = item.split("=", 1)
        override_slug, text = override_slug.strip(), text.strip()
        if not override_slug or not text:
            raise SystemExit(f"Invalid --subject-override {item!r}; expected non-empty SLUG=TEXT")
        subject_overrides[override_slug] = text
    for source in sources:
        if source["slug"] in subject_overrides:
            source["subject"] = subject_overrides[source["slug"]]
    if args.seed_offset:
        sources = [{**source, "seed": int(source["seed"]) + args.seed_offset} for source in sources]
    if len(sources) != 20:
        raise ValueError(f"Expected exactly 20 enabled source cases, found {len(sources)}")
    required = [PYTHON, AUTOMATION, ANCHOR_GATE, ANCHOR_CATALOG, SOURCE_CSV, SOURCE_GATE_CATALOG]
    required += [source[key] for source in sources for key in ("image", "lora")]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing formal assets:\n" + "\n".join(missing))

    repair_cases: dict[int, set[str]] = {}
    for item in args.repair_case:
        job_text, variant = item.split(":", 1)
        allowed_variants = (
            {"hybrid_carrier"}
            if args.construction_condition == "hybrid"
            else {"baseline", "qwen_anchor"}
        )
        if variant not in allowed_variants:
            raise ValueError(f"Unknown repair variant: {variant}")
        repair_cases.setdefault(int(job_text), set()).add(variant)

    pairs = []
    for source_index, source in enumerate(sources):
        for target_index, target in enumerate(TARGETS):
            job_id = source_index * len(TARGETS) + target_index
            if (repair_cases and job_id in repair_cases) or (
                not repair_cases and job_id % args.num_shards == args.shard_index
            ):
                pairs.append((job_id, source, target))
    total_pairs = len(sources) * len(TARGETS)
    expected = total_pairs // args.num_shards + (
        1 if args.shard_index < total_pairs % args.num_shards else 0
    )
    if not repair_cases and len(pairs) != expected:
        raise RuntimeError(f"Shard cardinality mismatch: {len(pairs)} != {expected}")

    output_base = args.output_base
    if args.construction_condition == "hybrid" and output_base == DEFAULT_OUTPUT_BASE:
        output_base = DEFAULT_HYBRID_OUTPUT_BASE
    output_root = output_base.expanduser().resolve() / f"worker_{args.shard_index:02d}"
    anchors = {}
    for target in TARGETS:
        selected = lookup_anchor(int(target["class"]), ANCHOR_CATALOG)
        if str(selected["target_label"]).lower() != target["label"].lower():
            raise ValueError(f"Anchor catalog mismatch: {selected}")
        anchors[target["label"]] = {
            "label": str(selected["anchor_label"]), "class": int(selected["anchor_class"])
        }
    hybrid_catalog = None
    if args.construction_condition == "hybrid":
        hybrid_catalog = load_hybrid_catalog(args.hybrid_prompt_catalog)
        validate_formal_mapping(
            TARGETS,
            hybrid_catalog,
            lambda target_class: lookup_anchor(target_class, ANCHOR_CATALOG),
        )

    plan = {
        "design": (
            "20 sources x 30 targets x supplemental Hybrid Carrier x CRA/JIA/CIRA"
            if args.construction_condition == "hybrid"
            else "20 DreamBooth sources x 30 fixed targets x baseline/anchor x CRA/JIA/CIRA"
        ),
        "construction_condition": args.construction_condition,
        "hybrid_prompt_catalog": str(args.hybrid_prompt_catalog) if hybrid_catalog else None,
        "num_shards": args.num_shards, "shard_index": args.shard_index,
        "attack_scale": args.attack_scale,
        "assigned_pair_count": len(pairs), "methods": METHODS,
        "disabled_sources": [],
        "sources": [{key: str(value) if isinstance(value, Path) else value for key, value in source.items()}
                    for source in sources],
        "targets": list(TARGETS),
        "assigned_pairs": [{"job_id": job_id, "source": source["slug"], **target}
                           for job_id, source, target in pairs],
        "status": "dry_run" if args.dry_run else "running", "jobs": [],
    }
    if args.dry_run:
        variants_per_pair = 1 if args.construction_condition == "hybrid" else 2
        if hybrid_catalog:
            previews = []
            for _, source, target in pairs:
                entry = hybrid_catalog[int(target["class"])]
                prompts = build_hybrid_prompts(
                    entry, f"[V] {source['subject']}", source["subject"]
                )
                previews.append({
                    "source": source["slug"], "target": target["label"],
                    "carrier": entry["carrier_label"], **prompts,
                })
            plan["hybrid_prompt_previews"] = previews
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"PREFLIGHT_OK | worker={args.shard_index} pairs={len(pairs)} variants={len(pairs) * variants_per_pair}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "experiment_manifest.json"
    env = dict(os.environ)
    env.update(HF_HOME="/root/autodl-tmp/hf_home", HF_HUB_OFFLINE="1",
               TRANSFORMERS_OFFLINE="1", TORCH_HOME="/root/.cache/torch")
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    for pair_number, (job_id, source, target) in enumerate(pairs, 1):
        anchor = anchors[target["label"]]
        if hybrid_catalog:
            entry = hybrid_catalog[int(target["class"])]
            prompt_spec = build_hybrid_prompts(
                entry, f"[V] {source['subject']}", source["subject"]
            )
            variants = (("hybrid_carrier", anchor["label"], anchor["class"], prompt_spec),)
        else:
            variants = (
                ("baseline", None, target["class"], None),
                ("qwen_anchor", anchor["label"], anchor["class"], None),
            )
        for variant, visible, gate_class, prompt_spec in variants:
            if repair_cases and variant not in repair_cases[job_id]:
                continue
            free_bytes = shutil.disk_usage(output_root).free
            if free_bytes < 10 * 1024 ** 3:
                raise RuntimeError(f"Stopping safely: only {free_bytes / 1024 ** 3:.2f} GiB free")
            run_name = f'{source["slug"]}_to_{slug(target["label"])}_{variant}'
            run_root = output_root / run_name
            job = {"job_id": job_id, "source": source["slug"], "target": target["label"],
                   "target_class": target["class"], "variant": variant,
                   "visible_object": visible or target["label"], "run_name": run_name,
                   "prompt_spec": prompt_spec}
            plan["jobs"].append(job)
            if (run_root / "summary.json").is_file():
                job["status"] = "complete_reused"
                continue
            if (run_root / ".gate_failed_skip").is_file():
                job["status"] = "failed_gate_preserved"
                continue
            if (run_root / ".prepare_failed_skip").is_file():
                job["status"] = "failed_prepare_preserved"
                continue
            if (run_root / ".attack_failed_skip").is_file():
                job["status"] = "failed_attack_preserved"
                continue
            job["status"] = "running"
            manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"PAIR {pair_number}/{len(pairs)} | job_id={job_id} | {run_name}", flush=True)
            source_gate_path = run_root / "source_gate.json"
            source_gate_record = (
                json.loads(source_gate_path.read_text(encoding="utf-8-sig"))
                if source_gate_path.is_file() else {}
            )
            if source_gate_record.get("pass") is not True:
                try:
                    execute(automation_command(
                                output_root, source, target, variant, visible, "prepare",
                                args.attack_scale, prompt_spec=prompt_spec),
                            output_root / f"{run_name}.prepare.log", False, env)
                except subprocess.CalledProcessError as error:
                    job.update(status="failed_prepare", prepare_returncode=error.returncode)
                    marker = run_root / ".prepare_failed_skip"
                    marker.write_text(
                        "Preserved failed preparation/source gate; do not retry during resume.\n",
                        encoding="utf-8",
                    )
                    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue
            gate_args = [str(PYTHON), str(ANCHOR_GATE), "--run-root", str(run_root),
                         "--anchor-label", visible or target["label"],
                         "--anchor-class", str(gate_class),
                         "--max-prompt-rounds", "3", "--seeds-per-prompt", "3"]
            gate_path = run_root / "qwen_gate.json"
            gate_record = json.loads(gate_path.read_text(encoding="utf-8-sig")) if gate_path.is_file() else {}
            if not (gate_record.get("pass") is True or gate_record.get("forced_best") is True):
                try:
                    execute(gate_args, output_root / f"{run_name}.gate.log", False, env)
                except subprocess.CalledProcessError as error:
                    job.update(status="failed_gate", gate_returncode=error.returncode)
                    (run_root / ".gate_failed_skip").write_text(
                        "Preserved failed gate; do not retry during resume.\n", encoding="utf-8")
                    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                    continue
            try:
                execute(automation_command(
                            output_root, source, target, variant, visible, "attack",
                            args.attack_scale, prompt_spec=prompt_spec),
                        output_root / f"{run_name}.attack.log", False, env)
            except subprocess.CalledProcessError as error:
                job.update(status="failed_attack_gate_or_method", attack_returncode=error.returncode)
                (run_root / ".attack_failed_skip").write_text(
                    "Preserved failed clean-inpainting gate or attack; do not retry during resume.\n",
                    encoding="utf-8",
                )
                manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
                continue
            job["status"] = "complete"
            manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")

    plan["status"] = "complete"
    manifest_path.write_text(json.dumps(plan, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FORMAL_SHARD_COMPLETE | worker={args.shard_index} pairs={len(pairs)}")


if __name__ == "__main__":
    main()
