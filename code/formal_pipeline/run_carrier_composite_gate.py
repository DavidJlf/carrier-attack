#!/usr/bin/env python3
"""Qwen-VL and ImageNet classifier gate for a carrier composite."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("FLUX_PYTHON", sys.executable))
QWEN = HERE / "qwen_composite_gate.py"
GENERATOR = HERE / "generate_composite.py"
MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-VL-7B-Instruct")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--carrier-label", required=True)
    parser.add_argument("--carrier-class", required=True, type=int)
    parser.add_argument("--max-prompt-rounds", type=int, default=3)
    parser.add_argument("--seeds-per-prompt", type=int, default=3)
    return parser.parse_args()


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    print("COMMAND |", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def snapshot_evaluated_candidate(run_root: Path, attempt: int, record: dict) -> Path:
    """Bind the image files that Qwen actually saw to its attempt record."""
    assets = run_root / "assets"
    snapshot = assets / "carrier_gate_evaluated" / f"attempt_{attempt:03d}"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in ("composite.png", "composite_GENERATED.png", "composite_MASK.png"):
        source = assets / name
        if source.is_file():
            shutil.copy2(source, snapshot / name)
    manifest = load(run_root / "manifest.json")
    save(snapshot / "selection_metadata.json", {
        "attempt": attempt,
        "background_prompt": manifest.get("background_prompt"),
        "qwen_target_phrase": manifest.get("qwen_target_phrase"),
        "seed": manifest.get("seed"),
        "qwen_record": record,
    })
    return snapshot


def failed_candidate_score(record: dict) -> tuple:
    """Rank failed candidates by Qwen quality, then classifier evidence."""
    evidence = record.get("classifier_evidence", {})
    rank = int(evidence.get("target_rank", 1001))
    probability = float(evidence.get("target_probability", 0.0))
    positive_fields = (
        "target_present", "correct_visual_identity", "correct_visual_subtype",
        "exactly_one_target", "recognizable_before_merging",
    )
    qwen_positive = sum(record.get(key) is True for key in positive_fields)
    no_severe_defect = sum(
        record.get(key) is False
        for key in ("severe_anatomical_defect", "severe_structural_defect")
    )
    return (
        record.get("target_present") is True,
        qwen_positive,
        no_severe_defect,
        -rank,
        probability,
    )


def restore_forced_best(run_root: Path, candidates: list[tuple[dict, Path]],
                        label: str, target_class: int) -> None:
    record, snapshot = max(candidates, key=lambda item: failed_candidate_score(item[0]))
    assets = run_root / "assets"
    for name in ("composite.png", "composite_GENERATED.png", "composite_MASK.png"):
        source = snapshot / name
        if source.is_file():
            shutil.copy2(source, assets / name)
    metadata = load(snapshot / "selection_metadata.json")
    attempt = int(record["attempt"])
    forced = {
        **record,
        "pass": False,
        "combined_gate_pass": False,
        "forced_best": True,
        "decision_source": "qwen_best_failed_candidate",
        "gate_target_label": label,
        "gate_target_class": target_class,
        "classifier_top10_required": True,
        "selected_snapshot": str(snapshot),
        "selection_score": list(failed_candidate_score(record)),
    }
    save(run_root / "qwen_gate.json", forced)
    save(run_root / "classifier_gate_normal.json", {
        **record.get("classifier_evidence", {}),
        "pass": False,
        "forced_best": True,
        "gate_target_label": label,
    })
    manifest_path = run_root / "manifest.json"
    manifest = load(manifest_path)
    for key in ("background_prompt", "qwen_target_phrase", "seed"):
        if metadata.get(key) is not None:
            manifest[key] = metadata[key]
    manifest["carrier_gate_status"] = "forced_best"
    manifest["carrier_gate_attempt"] = attempt
    manifest["carrier_gate_selected_snapshot"] = str(snapshot)
    save(manifest_path, manifest)
    evidence = record.get("classifier_evidence", {})
    print(
        f"CARRIER GATE FORCED BEST | {label} class={target_class} "
        f"attempt={attempt} rank={evidence.get('target_rank', 1001)} "
        f"prob={evidence.get('target_probability', 0.0)}",
        flush=True,
    )


def regenerate(run_root: Path, prompt: str, seed: int, phrase: str, label: str) -> None:
    manifest_path = run_root / "manifest.json"
    manifest = load(manifest_path)
    assets = run_root / "assets"
    archive = assets / "carrier_gate_attempts" / label
    archive.mkdir(parents=True, exist_ok=True)
    for name in ("composite.png", "composite_GENERATED.png", "composite_MASK.png"):
        path = assets / name
        if path.is_file():
            shutil.copy2(path, archive / name)
    save(archive / "generation.json", {
        "background_prompt": manifest["background_prompt"],
        "seed": manifest["seed"],
        "visible_carrier": manifest["visible_carrier"],
    })
    run([
        str(PYTHON), str(GENERATOR),
        "--source_image", manifest["copied_source"],
        "--subject_mask", str(assets / "subject_mask.png"),
        "--prompt", prompt,
        "--lora_path", manifest["lora_path"], "--lora_scale", str(manifest["lora_scale"]),
        "--num_steps", "30", "--guidance_scale", "3.5", "--seed", str(seed),
        "--height", str(manifest["composite_resolution"]),
        "--width", str(manifest["composite_resolution"]),
        "--mask_feather", "4", "--output", str(assets / "composite.png"),
        "--local_files_only",
    ])
    manifest["background_prompt"] = prompt
    manifest["qwen_target_phrase"] = phrase
    manifest["seed"] = seed
    save(manifest_path, manifest)


def main() -> None:
    args = parse_args()
    run_root = args.run_root.expanduser().resolve()
    manifest_path = run_root / "manifest.json"
    manifest = load(manifest_path)
    manifest["gate_target_label"] = args.carrier_label
    manifest["gate_target_class"] = args.carrier_class
    manifest["qwen_requested_target"] = args.carrier_label
    save(manifest_path, manifest)
    stale_gate = run_root / "qwen_gate.json"
    if stale_gate.is_file():
        stale_gate.unlink()

    attempt = 1
    base_seed = int(manifest["seed"])
    evaluated_candidates: list[tuple[dict, Path]] = []
    for prompt_round in range(1, args.max_prompt_rounds + 1):
        run([
            str(PYTHON), str(QWEN), "--model", str(MODEL),
            "--attempt", str(attempt), "--classifier-failed-advisor",
            "--run-roots", str(run_root),
        ])
        record_path = run_root / f"qwen_gate_attempt_{attempt}.json"
        record = load(record_path)
        evaluated_candidates.append((record, snapshot_evaluated_candidate(run_root, attempt, record)))
        rank = int(record.get("classifier_evidence", {}).get("target_rank", 1001))
        if record.get("pass") is True and rank <= 10:
            record.update({
                "pass": True,
                "combined_gate_pass": True,
                "gate_target_label": args.carrier_label,
                "gate_target_class": args.carrier_class,
                "classifier_top10_required": True,
            })
            save(stale_gate, record)
            save(run_root / "classifier_gate_normal.json", {
                **record["classifier_evidence"], "pass": True,
                "gate_target_label": args.carrier_label,
            })
            manifest = load(manifest_path)
            manifest["carrier_gate_status"] = "passed"
            manifest["carrier_gate_attempt"] = attempt
            save(manifest_path, manifest)
            print(f"CARRIER GATE PASS | {args.carrier_label} class={args.carrier_class} rank={rank}")
            return

        manifest = load(manifest_path)
        current = manifest["background_prompt"]
        revised = str(record.get("adjusted_prompt") or current).strip()
        phrase = str(record.get("adjusted_target_phrase") or manifest.get("qwen_target_phrase") or args.carrier_label).strip()
        attempt += 1
        regenerate(run_root, revised, base_seed, phrase, f"prompt_round_{prompt_round}_base_seed")
        for seed_trial in range(1, args.seeds_per_prompt + 1):
            run([
                str(PYTHON), str(QWEN), "--model", str(MODEL),
                "--attempt", str(attempt), "--classifier-failed-advisor",
                "--run-roots", str(run_root),
            ])
            record = load(run_root / f"qwen_gate_attempt_{attempt}.json")
            evaluated_candidates.append((record, snapshot_evaluated_candidate(run_root, attempt, record)))
            rank = int(record.get("classifier_evidence", {}).get("target_rank", 1001))
            if record.get("pass") is True and rank <= 10:
                record.update({"pass": True, "combined_gate_pass": True,
                               "gate_target_label": args.carrier_label,
                               "gate_target_class": args.carrier_class,
                               "classifier_top10_required": True})
                save(stale_gate, record)
                save(run_root / "classifier_gate_normal.json", {
                    **record["classifier_evidence"], "pass": True,
                    "gate_target_label": args.carrier_label,
                })
                manifest = load(manifest_path)
                manifest["carrier_gate_status"] = "passed"
                manifest["carrier_gate_attempt"] = attempt
                save(manifest_path, manifest)
                print(f"CARRIER GATE PASS | {args.carrier_label} class={args.carrier_class} rank={rank}")
                return
            seed = base_seed + prompt_round * 10000 + seed_trial * 1009 + args.carrier_class
            manifest = load(manifest_path)
            regenerate(run_root, manifest["background_prompt"], seed,
                       manifest.get("qwen_target_phrase", args.carrier_label),
                       f"prompt_round_{prompt_round}_seed_trial_{seed_trial}")
            attempt += 1
    if not evaluated_candidates:
        raise RuntimeError(f"Anchor gate produced no evaluated candidates for {run_root}")
    restore_forced_best(run_root, evaluated_candidates, args.carrier_label, args.carrier_class)


if __name__ == "__main__":
    main()
