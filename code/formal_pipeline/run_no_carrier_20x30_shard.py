#!/usr/bin/env python3
"""Run one deterministic shard of the matched 20x30 No-Carrier experiment."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

from run_formal_17x30_shard import TARGETS, load_sources


HERE = Path(__file__).resolve().parent
FLUX_PYTHON = Path("/root/autodl-tmp/venvs/flux/bin/python")
RETURN_RUNNER = HERE / "run_cra.py"
FLUX_ATTACK = HERE / "flux_attack.py"
TRANSFER_EVAL = HERE / "evaluate_transferability.py"
DEFAULT_OUTPUT = Path(
    "/root/autodl-tmp/FLUX_FORMAL_MASTER/experiments/formal_no_carrier_20x30"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-shards", type=int, default=5)
    parser.add_argument("--shard-index", type=int, required=True)
    parser.add_argument("--output-base", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--attack-intervals", type=int, default=15)
    parser.add_argument("--attack-scale", type=float, default=0.5)
    parser.add_argument(
        "--reuse-clean-base", type=Path,
        help="Existing formal No-Carrier root; reuses each matched 768px input.png.",
    )
    parser.add_argument(
        "--reuse-clean-run-name", default="matched_scale0p5",
        help="Child run directory that contains input.png under each reusable case.",
    )
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="Skip auxiliary black-box transfer evaluation for preservation-only runs.",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def target_slug(label: str) -> str:
    return "_".join(label.lower().replace("-", " ").split())


def prompt_path(image: Path) -> Path:
    text = str(image)
    candidates = [
        Path(text.replace("/quality/selected_top40/", "/quality/selected_top40_prompts/")).with_suffix(".txt"),
        Path(text.replace("/images/", "/prompts/")).with_suffix(".txt"),
        image.with_suffix(".txt"),
    ]
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"No exact source prompt found for {image}")


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8-sig"))


def write_json_atomic(path: Path, value) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_logged(command: list[str], log_path: Path, env: dict[str, str]) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("COMMAND: " + " ".join(command) + "\n")
        handle.flush()
        subprocess.run(command, check=True, stdout=handle, stderr=subprocess.STDOUT, env=env)


def main() -> None:
    args = parse_args()
    if args.num_shards < 1 or not 0 <= args.shard_index < args.num_shards:
        raise ValueError("Require num_shards >= 1 and 0 <= shard_index < num_shards")
    if not 1 <= args.attack_intervals < 50:
        raise ValueError("Require 1 <= attack_intervals < 50")
    if args.attack_scale <= 0:
        raise ValueError("Require --attack-scale > 0")
    sources = load_sources()
    if len(sources) != 20 or len(TARGETS) != 30:
        raise RuntimeError(f"Expected 20x30, found {len(sources)}x{len(TARGETS)}")
    required = [FLUX_PYTHON, RETURN_RUNNER, FLUX_ATTACK, TRANSFER_EVAL]
    for source in sources:
        required.extend([source["image"], source["lora"], prompt_path(source["image"])])
    missing = [str(path) for path in required if not Path(path).exists()]
    if missing:
        raise FileNotFoundError("Missing required assets:\n" + "\n".join(missing))

    all_pairs = [
        (source_index * len(TARGETS) + target_index, source, target)
        for source_index, source in enumerate(sources)
        for target_index, target in enumerate(TARGETS)
    ]
    pairs = [item for item in all_pairs if item[0] % args.num_shards == args.shard_index]
    expected = len(all_pairs) // args.num_shards + (
        1 if args.shard_index < len(all_pairs) % args.num_shards else 0
    )
    if len(pairs) != expected:
        raise RuntimeError(f"Shard cardinality mismatch: {len(pairs)} != {expected}")

    output_root = args.output_base.expanduser().resolve() / f"worker_{args.shard_index:02d}"
    plan = {
        "design": "matched No-Carrier: same 20 sources x same 30 targets",
        "configuration": {
            "route": "single-image FLUX finite return",
            "progress": f"50->{50 - args.attack_intervals}->50",
            "attack_intervals": args.attack_intervals,
            "classifier_scale": args.attack_scale,
            "global_attack": True,
            "subject_mask": False,
            "guidance_scale": 2.5,
            "inversion_guidance_scale": 2.5,
            "recon_guidance_scale": 1.0,
            "pivot_correction": 1.0,
            "resolution": 768,
        },
        "num_shards": args.num_shards,
        "shard_index": args.shard_index,
        "assigned_pair_count": len(pairs),
        "assigned_pairs": [
            {
                "job_id": job_id,
                "source": source["slug"],
                "source_image": str(source["image"]),
                "source_lora": str(source["lora"]),
                "source_seed": source["seed"],
                "source_classes": source["source_classes"],
                "prompt_file": str(prompt_path(source["image"])),
                "target_class": target["class"],
                "target_label": target["label"],
                "target_group": target["group"],
            }
            for job_id, source, target in pairs
        ],
        "jobs": [],
        "status": "dry_run" if args.dry_run else "running",
    }
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        print(f"PREFLIGHT_OK | worker={args.shard_index} pairs={len(pairs)}")
        return

    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "experiment_manifest.json"
    write_json_atomic(manifest_path, plan)
    env = dict(os.environ)
    env.update(
        HF_HOME="/root/autodl-tmp/hf_home",
        HF_HUB_OFFLINE="1",
        TRANSFORMERS_OFFLINE="1",
        TORCH_HOME="/root/.cache/torch",
        PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True",
    )

    for ordinal, (job_id, source, target) in enumerate(pairs, 1):
        if shutil.disk_usage(output_root).free < 8 * 1024**3:
            raise RuntimeError("Stopping safely: less than 8 GiB free")
        name = f'{source["slug"]}_to_{target_slug(target["label"])}_no_carrier'
        job_root = output_root / name
        summary_path = job_root / "summary.json"
        job = {
            "job_id": job_id,
            "source": source["slug"],
            "target_class": target["class"],
            "target_label": target["label"],
            "run_name": name,
        }
        plan["jobs"].append(job)
        if summary_path.is_file() and read_json(summary_path).get("status") == "complete":
            job["status"] = "complete_reused"
            write_json_atomic(manifest_path, plan)
            continue
        job_root.mkdir(parents=True, exist_ok=True)
        prompt_file = prompt_path(source["image"])
        prompt = prompt_file.read_text(encoding="utf-8-sig").strip()
        input_image = source["image"]
        if args.reuse_clean_base:
            previous = list(args.reuse_clean_base.glob(f"worker_*/{name}/{args.reuse_clean_run_name}"))
            if len(previous) != 1 or not (previous[0] / "input.png").is_file():
                raise FileNotFoundError(f"Expected one reusable clean input for {name}, found {previous}")
            input_image = previous[0] / "input.png"
            old_summary = previous[0].parent / "summary.json"
            if old_summary.is_file():
                prompt = read_json(old_summary).get("source_prompt", prompt)
        run_name = f"scale{args.attack_scale:g}_late{args.attack_intervals}"
        command = [
            str(FLUX_PYTHON), str(RETURN_RUNNER),
            "--base-flux-attack", str(FLUX_ATTACK),
            "--full-steps", "50", "--return-to-progress", str(50 - args.attack_intervals),
            "--run-name", run_name, "--output-root", str(job_root),
            "--input_image", str(input_image), "--single_image_attack",
            "--prompt", prompt, "--inversion_prompt", prompt,
            "--target_class", str(target["class"]), "--target_label", target["label"],
            "--source_classes", source["source_classes"],
            "--guidance_scale", "2.5", "--inversion_guidance_scale", "2.5",
            "--recon_guidance_scale", "1.0", "--pivot_correction", "1.0",
            "--classifier_scale", str(args.attack_scale), "--normalize_grad",
            "--objective", "ce", "--attack_sign", "1",
            "--height", "768", "--width", "768", "--seed", str(source["seed"]),
            "--local_files_only", "--skip_clean_baseline",
            "--lora_path", str(source["lora"]), "--lora_scale", "1.0",
        ]
        print(
            f"PAIR {ordinal}/{len(pairs)} | job_id={job_id} | {name}",
            flush=True,
        )
        job["status"] = "attack_running"
        write_json_atomic(manifest_path, plan)
        try:
            run_logged(command, job_root / "attack.log", env)
            method_root = job_root / run_name
            attack_eval_path = method_root / "attack_eval.json"
            images = sorted((method_root / "images").glob("*ATTACK*.png"))
            if not attack_eval_path.is_file() or len(images) != 1:
                raise RuntimeError(
                    f"Expected one attack image and attack_eval.json, found {len(images)}"
                )
            if args.skip_transfer:
                black_box = {"status": "skipped", "reason": "preservation-only run"}
            else:
                transfer_json = job_root / "transfer_eval.json"
                transfer_csv = job_root / "transfer_eval.csv"
                transfer_command = [
                    str(FLUX_PYTHON), str(TRANSFER_EVAL),
                    "--image", str(images[0]), "--target-class", str(target["class"]),
                    "--models", "resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b",
                    "--output-json", str(transfer_json), "--output-csv", str(transfer_csv),
                ]
                job["status"] = "transfer_running"
                write_json_atomic(manifest_path, plan)
                run_logged(transfer_command, job_root / "transfer.log", env)
                black_box = read_json(transfer_json)
            summary = {
                "status": "complete",
                "condition": "no_carrier",
                "job_id": job_id,
                "source": {key: str(value) if isinstance(value, Path) else value for key, value in source.items()},
                "source_prompt": prompt,
                "source_prompt_file": str(prompt_file),
                "target": target,
                "configuration": plan["configuration"],
                "attack_image": str(images[0]),
                "white_box": read_json(attack_eval_path),
                "black_box": black_box,
            }
            write_json_atomic(summary_path, summary)
            job["status"] = "complete"
        except Exception as error:
            job["status"] = "failed"
            job["error"] = repr(error)
            write_json_atomic(job_root / "failure.json", job)
        write_json_atomic(manifest_path, plan)

    complete = sum(job.get("status", "").startswith("complete") for job in plan["jobs"])
    plan["status"] = "complete" if complete == len(pairs) else "completed_with_failures"
    plan["complete_count"] = complete
    plan["failed_count"] = len(pairs) - complete
    write_json_atomic(manifest_path, plan)
    print(
        f"NO_CARRIER_SHARD_DONE | worker={args.shard_index} "
        f"complete={complete} failed={len(pairs) - complete}",
        flush=True,
    )


if __name__ == "__main__":
    main()
