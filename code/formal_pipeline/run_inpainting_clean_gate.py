#!/usr/bin/env python3
"""Generate and gate a clean inpainting before an attack is allowed to run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
from PIL import Image, ImageFilter, ImageStat

HERE = Path(__file__).resolve().parent
PYTHON = Path("/root/autodl-tmp/venvs/flux/bin/python")
TEACHER = HERE / "flux_kontext_inpaint_attack.py"
QWEN = HERE / "qwen_composite_gate.py"
EVALUATE = HERE / "evaluate_imagenet.py"
MODEL = Path("/root/autodl-tmp/models/Qwen2.5-VL-7B-Instruct")
SAM_PYTHON = Path("/root/autodl-tmp/sam3/env/bin/python")
SAM_SCRIPT = HERE / "sam3_single.py"


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def save(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def run(command: list[str]) -> None:
    print("COMMAND |", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def prompt_for(phrase: str) -> str:
    return (
        "Edit only the masked background. Preserve the existing foreground subject unchanged. "
        f"In the distant background, exactly one {phrase}, slightly blurred by shallow depth of field, "
        "with visible edges. Do not add or modify any foreground subject. Do not duplicate the original "
        "foreground subject and do not create a second foreground subject."
    )


def make_background_only(image_path: Path, mask_path: Path, output_path: Path) -> None:
    image = Image.open(image_path).convert("RGB")
    mask = Image.open(mask_path).convert("L").resize(image.size)
    # Dilate the SAM3 mask to remove fur/boundary evidence as well as the core cat.
    mask = mask.filter(ImageFilter.MaxFilter(21)).point(lambda value: 255 if value >= 128 else 0)
    inverse = Image.eval(mask, lambda value: 255 - value)
    stat = ImageStat.Stat(image, mask=inverse)
    fill = tuple(int(round(value)) for value in stat.mean[:3]) if stat.count[0] else (127, 127, 127)
    Image.composite(Image.new("RGB", image.size, fill), image, mask).save(output_path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--route", required=True)
    parser.add_argument("--max-attempts", type=int, default=4,
                        help="Initial clean plus at most three seed changes.")
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    output_dir = args.output_dir.resolve()
    manifest_path = run_root / "manifest.json"
    manifest = load(manifest_path)
    anchor = str(manifest["gate_target_label"])
    anchor_class = int(manifest["gate_target_class"])
    phrase = str(manifest.get("effective_visible_anchor") or anchor)
    current_prompt = prompt_for(phrase)
    base_seed = int(manifest["seed"])
    attempts_root = output_dir / "clean_gate_attempts"

    max_attempts = min(max(args.max_attempts, 1), 4)
    best: dict | None = None
    for attempt in range(1, max_attempts + 1):
        seed = base_seed if attempt == 1 else base_seed + attempt * 1009 + anchor_class
        attempt_dir = attempts_root / f"attempt_{attempt}"
        run([
            str(PYTHON), str(TEACHER), "--image", manifest["copied_source"],
            "--mask", str(run_root / "assets" / "background_mask.png"),
            "--inpaint-prompt", current_prompt, "--flux-steps", "50",
            "--seed", str(seed), "--output-dir", str(attempt_dir),
            "--no-cpu-offload",
        ])
        image = attempt_dir / "inpainted.png"
        sam_dir = attempt_dir / "sam3_subject"
        sam_command = [
            str(SAM_PYTHON), str(SAM_SCRIPT), "--image", str(image),
            "--prompt", str(manifest["subject"]), "--output-dir", str(sam_dir),
            "--merge-multiple",
        ]
        print("COMMAND |", " ".join(sam_command), flush=True)
        sam_result = subprocess.run(sam_command, check=False)
        subject_mask = sam_dir / "subject_mask.png"
        subject_detected = sam_result.returncode == 0 and subject_mask.is_file()
        if not subject_detected:
            print(
                f"SAM3 SUBJECT MISS | attempt={attempt} subject={manifest['subject']} | "
                "candidate remains eligible only for forced fallback",
                flush=True,
            )
            subject_mask = run_root / "assets" / "subject_mask.png"
            if not subject_mask.is_file():
                raise FileNotFoundError(f"Fallback subject mask missing: {subject_mask}")
        background_only = attempt_dir / "background_only.png"
        make_background_only(image, subject_mask, background_only)
        manifest = load(manifest_path)
        manifest["teacher_prompt"] = current_prompt
        save(manifest_path, manifest)
        suffix = f"{args.route}_clean"
        if attempt <= 2:
            run([
                str(PYTHON), str(QWEN), "--model", str(MODEL), "--attempt", str(attempt),
                "--classifier-failed-advisor", "--inpainting-result",
                "--image-path", str(image), "--prompt-key", "teacher_prompt",
                "--classifier-image-path", str(background_only),
                "--record-suffix", suffix, "--run-roots", str(run_root),
            ])
            record_path = run_root / f"qwen_gate_{suffix}_attempt_{attempt}.json"
            record = load(record_path)
        else:
            record_path = attempt_dir / "classifier_only.json"
            run([
                str(PYTHON), str(EVALUATE), "--image", str(background_only),
                "--target-class", str(anchor_class), "--output", str(record_path),
            ])
            evidence = load(record_path)
            rank_only = int(evidence.get("target_rank", 1001))
            record = {
                "pass": rank_only <= 10, "classifier_evidence": evidence,
                "decision_source": "classifier_only_after_qwen_limit",
                "qwen_invoked": False,
            }
        rank = int(record.get("classifier_evidence", {}).get("target_rank", 1001))
        candidate = {
            "rank": rank, "image": image, "background_only": background_only,
            "subject_mask": subject_mask, "prompt": current_prompt, "seed": seed,
            "attempt": attempt, "record_path": record_path,
            "subject_detected": subject_detected,
            "decision_source": record.get("decision_source"),
            "qwen_invoked": record.get("qwen_invoked", attempt <= 2),
        }
        if best is None or (not subject_detected, rank) < (
            not bool(best["subject_detected"]), int(best["rank"])
        ):
            best = candidate
        if subject_detected and record.get("pass") is True and rank <= 10:
            output_dir.mkdir(parents=True, exist_ok=True)
            shutil.copy2(image, output_dir / "clean.png")
            approval = {
                "pass": True, "route": args.route, "attempt": attempt,
                "anchor": anchor, "anchor_class": anchor_class,
                "target_rank": rank, "prompt": current_prompt, "seed": seed,
                "classifier_image": str(background_only),
                "sam3_subject_mask": str(subject_mask),
                "sam3_subject_detected": True,
                "decision_source": record.get("decision_source"),
                "qwen_invoked": record.get("qwen_invoked", True),
                "record": str(record_path),
            }
            save(output_dir / "clean_gate.json", approval)
            print(f"INPAINTING CLEAN GATE PASS | route={args.route} target={anchor} rank={rank}", flush=True)
            return
        # Qwen may revise the target phrase once only. All three subsequent
        # trials vary the seed under this single approved rewrite.
        if attempt == 1:
            phrase = str(record.get("adjusted_target_phrase") or phrase).strip()
            current_prompt = prompt_for(phrase)

    # After the bounded retry budget, continue with the best-ranked clean and
    # record that this was a forced fallback rather than a normal Top-10 pass.
    assert best is not None
    output_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best["image"], output_dir / "clean.png")
    approval = {
        "pass": True, "forced_fallback": True, "route": args.route,
        "attempt": best["attempt"], "attempts_exhausted": max_attempts,
        "anchor": anchor, "anchor_class": anchor_class,
        "target_rank": best["rank"], "prompt": best["prompt"], "seed": best["seed"],
        "classifier_image": str(best["background_only"]),
        "sam3_subject_mask": str(best["subject_mask"]),
        "sam3_subject_detected": bool(best["subject_detected"]),
        "decision_source": "bounded_inpainting_gate_fallback_best_rank",
        "qwen_rejection_limit": 2, "prompt_rewrite_limit": 1, "seed_change_limit": 3,
        "record": str(best["record_path"]),
    }
    save(output_dir / "clean_gate.json", approval)
    print(
        f"INPAINTING CLEAN GATE FORCED CONTINUE | route={args.route} "
        f"target={anchor} best_rank={best['rank']} after={max_attempts}", flush=True,
    )


if __name__ == "__main__":
    main()
