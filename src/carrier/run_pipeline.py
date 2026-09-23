#!/usr/bin/env python3
"""Automate SAM3, carrier construction, and CRA/CIRA/JIA.

The program is deliberately sequential so the four large FLUX jobs never
compete for the same GPU.  Originals are copied into a new run directory; no
source image, shared mask, model, or LoRA file is modified.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet50_Weights


HERE = Path(__file__).resolve().parent
TEACHER_ORIGINAL = HERE / "inpainting_attack.py"
INPAINTING_CLEAN_GATE = HERE / "run_inpainting_clean_gate.py"
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_SAM_PYTHON = Path(sys.executable)
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-image carrier construction plus CRA/CIRA/JIA."
    )
    parser.add_argument("--image", required=True, help="Input subject image.")
    parser.add_argument("--target", required=True, help="Exact ImageNet target label.")
    parser.add_argument(
        "--visible-anchor",
        help=(
            "Optional visible object used in every generation/inversion prompt. "
            "The classifier attack objective remains --target/--target-class."
        ),
    )
    parser.add_argument(
        "--target-class",
        type=int,
        help="Optional ImageNet index. If omitted, exact --target is resolved automatically.",
    )
    parser.add_argument(
        "--subject",
        required=True,
        help="Foreground subject label used by SAM3 and attack prompts, e.g. cat or dog.",
    )
    parser.add_argument(
        "--source-case",
        required=True,
        help=(
            "DreamBooth source-case slug used by the Top-10 source preservation "
            "gate, e.g. cat2 or grey_sloth_plushie."
        ),
    )
    parser.add_argument(
        "--concept-token",
        default="[V]",
        help="Learned token only. It is automatically combined with --subject; use an empty string if none.",
    )
    parser.add_argument(
        "--source-classes",
        required=True,
        help="Comma-separated ImageNet source classes corresponding to --subject.",
    )
    parser.add_argument(
        "--lora-path",
        required=True,
        help="FLUX LoRA corresponding to --subject.",
    )
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument(
        "--attack-scale",
        type=float,
        default=0.5,
        help="Classifier-gradient scale shared by CRA, JIA, and CIRA.",
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--run-name", help="Default: IMAGE_STEM_SUBJECT_TARGET.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--attack-resolution", type=int, default=768)
    parser.add_argument("--composite-resolution", type=int, default=1024)
    parser.add_argument("--sam-python", default=str(DEFAULT_SAM_PYTHON))
    parser.add_argument(
        "--phase", choices=("full", "prepare", "attack"), default="full",
        help=(
            "full runs preparation and attacks; prepare stops after composite "
            "generation; attack resumes an existing prepared run after Qwen passes."
        ),
    )
    parser.add_argument(
        "--background-prompt",
        help="Controlled composite-prompt override used only by Qwen retry rounds.",
    )
    parser.add_argument(
        "--teacher-prompt",
        help="Optional clean/JIA/CIRA inpainting-prompt override.",
    )
    parser.add_argument(
        "--attack-prompt",
        help="Optional CRA inversion/finite-return scene-prompt override.",
    )
    parser.add_argument(
        "--methods",
        default="cra,jia,cira",
        help=(
            "Comma-separated methods to run. Choices: cra, jia, cira. "
            "Any non-empty subset may be selected."
        ),
    )
    parser.add_argument(
        "--transfer-models", nargs="+",
        choices=("resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b"),
        default=("resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b"),
        help=(
            "Inference-only transfer models. New runs include ResNet-101 by default."
        ),
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_target(label: str, explicit_class: int | None) -> tuple[int, str]:
    categories = ResNet50_Weights.DEFAULT.meta["categories"]
    if explicit_class is not None:
        if explicit_class < 0 or explicit_class >= len(categories):
            raise ValueError(f"target class must be in [0, {len(categories) - 1}]")
        return explicit_class, categories[explicit_class]
    exact = [index for index, category in enumerate(categories) if category.lower() == label.lower()]
    if len(exact) != 1:
        partial = [category for category in categories if label.lower() in category.lower()]
        raise ValueError(
            f"Cannot uniquely resolve target {label!r}. Exact match required. "
            f"Partial matches: {partial[:20]}"
        )
    return exact[0], categories[exact[0]]


def article_for(label: str) -> str:
    return "an" if label[:1].lower() in "aeiou" else "a"


def subject_phrase(concept_token: str, subject: str) -> str:
    """Build the foreground concept without ever hard-coding a class name."""
    token = concept_token.strip()
    name = subject.strip()
    if not name:
        raise ValueError("--subject cannot be empty")
    return f"{token} {name}".strip()


def run_logged(
    name: str,
    command: list[str],
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
    method_number: int | None = None,
    method_total: int = 3,
    method_output: Path | None = None,
) -> int:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if method_number is None:
        print(f"\n{'-' * 96}")
        print(f"PREPARATION | {name}")
        print(f"{'-' * 96}")
    else:
        print(f"\n{'=' * 96}")
        print(f"METHOD {method_number}/{method_total} | {name}")
        if method_output is not None:
            print(f"RESULT DIRECTORY | {method_output}")
        print(f"{'=' * 96}")
    print(" ".join(command))
    if dry_run:
        if method_number is not None:
            print(f"METHOD {method_number}/{method_total} DRY-RUN COMPLETE | {name}")
        return 0
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            env=env,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            log.write(line)
            log.flush()
        code = process.wait()
    if method_number is not None:
        status = "SUCCESS" if code == 0 else f"FAILED (exit={code})"
        print(f"{'=' * 96}")
        print(f"METHOD {method_number}/{method_total} {status} | {name}")
        if method_output is not None:
            print(f"RESULT DIRECTORY | {method_output}")
        print(f"{'=' * 96}")
    return code


def read_json(path: Path) -> dict | None:
    if not path.is_file():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_transfer_summary(stages: list[dict], output: Path) -> None:
    """Combine existing per-method black-box results without rerunning models."""
    rows = []
    for stage in stages:
        payload = read_json(Path(stage["output"]) / "transfer_eval.json")
        if not payload:
            continue
        for result in payload.get("models", []):
            rows.append({"method": stage["method"], **result})
    if not rows:
        return
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_top1_gradcam(
    flux_python: Path,
    method_dir: Path,
    subject_mask: Path,
    log_path: Path,
    env: dict[str, str],
    dry_run: bool,
) -> tuple[int, dict | None]:
    """Explain the classifier's predicted Top-1 class after attacked.png exists."""
    attacked = method_dir / "attacked.png"
    output = method_dir / "gradcam_top1"
    if not dry_run and not attacked.is_file():
        raise FileNotFoundError(f"Grad-CAM input was not created: {attacked}")
    code = run_logged(
        "PREDICTED TOP-1 GRAD-CAM | attacked image",
        [
            str(flux_python), str(HERE / "gradcam_resnet50.py"),
            "--image", str(attacked),
            "--subject-mask", str(subject_mask),
            "--output-dir", str(output),
        ],
        log_path, env, dry_run,
    )
    if code != 0:
        raise RuntimeError(f"Grad-CAM failed for {method_dir} with exit code {code}")
    return code, None if dry_run else read_json(output / "gradcam.json")


def step_summary(path: Path) -> dict:
    if not path.is_file():
        return {"path": str(path), "count": 0}
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return {
        "path": str(path),
        "count": len(rows),
        "first_target_probability": rows[0].get("target_probability") if rows else None,
        "last_target_probability": rows[-1].get("target_probability") if rows else None,
    }


def contact_font(size: int, bold: bool = False):
    try:
        return ImageFont.truetype(
            "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf", size
        )
    except OSError:
        return ImageFont.load_default()


def export_run_contact(
    run_root: Path,
    run_name: str,
    source: Path,
    assets: Path,
    stages: list[dict],
    selected_methods: tuple[str, ...],
    inpainting_return_preparation_root: Path,
    teacher_root: Path,
    attack_scale: float,
) -> Path:
    """Export one run contact with white-box, Grad-CAM, and transfer results."""
    contacts = run_root / "contacts"
    contacts.mkdir(parents=True, exist_ok=True)
    # The extra caption height keeps four black-box transfer rows legible.
    tile, cap, gap, title_h = 390, 176, 18, 64
    columns = 1 + len(stages)
    canvas = Image.new(
        "RGB",
        (gap * (columns + 1) + tile * columns, title_h + gap * 3 + (tile + cap) * 2),
        "white",
    )
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, title_h // 2),
        f"{run_name} | methods: {', '.join(selected_methods)}",
        font=contact_font(28, True), fill="black", anchor="lm",
    )

    context_path = None
    context_label = None
    generated = assets / "composite_GENERATED.png"
    independent_clean = inpainting_return_preparation_root / f"aligned_rms{attack_scale:g}_late35_49" / "clean.png"
    teacher_clean = teacher_root / f"aligned_rms{attack_scale:g}_late35_49" / "clean.png"
    if "cra" in selected_methods and generated.is_file():
        context_path, context_label = generated, "Composite generated background"
    elif "cira" in selected_methods and independent_clean.is_file():
        context_path, context_label = independent_clean, "Independent masked clean inpainting"
    elif teacher_clean.is_file():
        context_path, context_label = teacher_clean, "Teacher clean inpainting"

    cells = [(0, 0, source, "Original source", None, None, None)]
    if context_path is not None:
        cells.append((1, 0, context_path, context_label, None, None, None))
    for col, stage in enumerate(stages, 1):
        method_dir = Path(stage["output"])
        evaluation = stage.get("final_eval")
        cam = stage.get("gradcam")
        transfer = stage.get("transfer_eval")
        cells.append((0, col, method_dir / "attacked.png", stage["method"], evaluation, None, transfer))
        cells.append((1, col, method_dir / "gradcam_top1/gradcam_overlay.png", "Predicted Top-1 Grad-CAM", evaluation, cam, None))

    for row, col, path, label, evaluation, cam, transfer in cells:
        if not path.is_file():
            raise FileNotFoundError(f"Contact input missing: {path}")
        x = gap + col * (tile + gap)
        y = title_h + gap + row * (tile + cap + gap)
        image = Image.open(path).convert("RGB")
        image.thumbnail((tile, tile), Image.Resampling.LANCZOS)
        panel = Image.new("RGB", (tile, tile), (238, 238, 238))
        panel.paste(image, ((tile - image.width) // 2, (tile - image.height) // 2))
        canvas.paste(panel, (x, y))
        draw.text((x, y + tile + 4), label, font=contact_font(19, True), fill="black")
        if evaluation is not None and cam is None:
            draw.text((x, y + tile + 32), f"Target P={100*evaluation['target_probability']:.2f}% Rank={evaluation['target_rank']}", font=contact_font(16), fill=(20, 55, 120))
            transfer_models = [] if not transfer else transfer.get("models", [])
            short_names = {
                "resnet101": "ResNet-101",
                "vgg19": "VGG19", "inception_v3": "IncV3",
                "convnext_base": "ConvNeXt-B", "swin_b": "Swin-B",
            }
            for transfer_index, result in enumerate(transfer_models[:5]):
                name = short_names.get(str(result.get("model")), str(result.get("model")))
                success = "YES" if result.get("targeted_success") else "no"
                line = (
                    f"{name}: P={100*float(result['target_probability']):.2f}% "
                    f"R={int(result['target_rank'])} ASR={success}"
                )
                draw.text(
                    (x, y + tile + 57 + transfer_index * 23), line,
                    font=contact_font(14), fill=(55, 55, 55),
                )
        if cam is not None:
            draw.text((x, y + tile + 32), f"Top-1: {cam['explained_label']} P={100*cam['explained_probability']:.2f}%", font=contact_font(16), fill=(20, 55, 120))
            fg, bg = cam.get("cam_energy_foreground_fraction"), cam.get("cam_energy_background_fraction")
            if fg is not None and bg is not None:
                draw.text((x, y + tile + 57), f"CAM FG={100*fg:.1f}% BG={100*bg:.1f}%", font=contact_font(16), fill=(120, 45, 20))

    output = contacts / f"{run_name}_contact.jpg"
    canvas.save(output, quality=94, subsampling=0)
    print(f"CONTACT EXPORTED | {output}", flush=True)
    return output


def export_transfer_contact(
    run_root: Path,
    run_name: str,
    stages: list[dict],
) -> Path | None:
    """Export a metrics-only white-box/black-box transferability table."""
    transfer_names = {
        str(item.get("model"))
        for stage in stages
        for item in (stage.get("transfer_eval") or {}).get("models", [])
    }
    model_order = tuple(
        name for name in
        ("resnet50", "resnet101", "vgg19", "inception_v3", "convnext_base", "swin_b")
        if name == "resnet50" or name in transfer_names
    )
    model_labels = {
        "resnet50": "ResNet-50 (white-box)",
        "resnet101": "ResNet-101 (black-box)",
        "vgg19": "VGG-19", "inception_v3": "Inception-v3",
        "convnext_base": "ConvNeXt-B", "swin_b": "Swin-B",
    }
    if not stages or not any(stage.get("transfer_eval") for stage in stages):
        return None
    contacts = run_root / "contacts"
    contacts.mkdir(parents=True, exist_ok=True)
    cell_w, row_h, gap, title_h, row_label_w = 360, 108, 12, 82, 225
    width = row_label_w + gap * (len(stages) + 1) + cell_w * len(stages)
    height = title_h + gap * (len(model_order) + 1) + row_h * len(model_order)
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text(
        (gap, title_h // 2),
        f"{run_name} | fixed ResNet-50 attacked image | transferability metrics",
        font=contact_font(26, True), fill="black", anchor="lm",
    )
    for col, stage in enumerate(stages):
        x = row_label_w + gap + col * (cell_w + gap)
        draw.text((x, title_h - 8), stage["method"], font=contact_font(17, True), fill="black", anchor="ls")
        results = {
            str(item.get("model")): item
            for item in (stage.get("transfer_eval") or {}).get("models", [])
        }
        white_box = stage.get("final_eval") or {}
        if white_box:
            results["resnet50"] = {
                **white_box,
                "targeted_success": int(white_box.get("target_rank", 1001)) == 1,
            }
        for row, model_name in enumerate(model_order):
            y = title_h + gap + row * (row_h + gap)
            if col == 0:
                draw.text((gap, y + row_h // 2), model_labels[model_name], font=contact_font(17, True), fill="black", anchor="lm")
            result = results.get(model_name)
            if not result:
                continue
            success = "YES" if result.get("targeted_success") else "no"
            fill = (224, 246, 227) if result.get("targeted_success") else (242, 242, 242)
            draw.rounded_rectangle((x, y, x + cell_w, y + row_h), radius=9, fill=fill, outline=(180, 180, 180), width=1)
            draw.text((x + 12, y + 13), f"Target P={100*float(result['target_probability']):.3f}%  Rank={int(result['target_rank'])}", font=contact_font(16, True), fill=(20, 55, 120))
            draw.text((x + 12, y + 44), f"Top-1: {result.get('top1_label', 'n/a')}", font=contact_font(15), fill=(35, 35, 35))
            draw.text((x + 12, y + 73), f"Targeted success: {success}", font=contact_font(15), fill=(100, 45, 20))
    output = contacts / f"{run_name}_transferability_contact.jpg"
    canvas.save(output, quality=94, subsampling=0)
    print(f"TRANSFERABILITY CONTACT EXPORTED | {output}", flush=True)
    return output


def extract_teacher_step_probabilities(log_path: Path, target_label: str, output: Path) -> None:
    """Extract the teacher script's printed target probabilities without changing it."""
    if not log_path.is_file():
        return
    pattern = re.compile(
        rf"\[attack\] step\s+(\d+).*?{re.escape(target_label)}\s+\(([0-9.eE+-]+)\)"
    )
    rows = []
    for match in pattern.finditer(log_path.read_text(encoding="utf-8", errors="replace")):
        step = int(match.group(1))
        if 30 <= step <= 49:
            rows.append(
                {"step": step, "target_probability": match.group(2),
                 "source": "teacher_original_top3_log_rounded"}
            )
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    valid_methods = (
        "cra",
        "jia",
        "cira",
    )
    selected_methods = tuple(
        item.strip()
        for item in args.methods.split(",") if item.strip()
    )
    if not selected_methods:
        raise ValueError("--methods must select at least one method")
    unknown_methods = sorted(set(selected_methods) - set(valid_methods))
    if unknown_methods:
        raise ValueError(f"Unknown --methods entries: {unknown_methods}")
    if len(set(selected_methods)) != len(selected_methods):
        raise ValueError(f"Duplicate --methods entries: {selected_methods}")
    # Always execute in the canonical method order regardless of CLI ordering.
    selected_methods = tuple(name for name in valid_methods if name in selected_methods)
    method_total = len(selected_methods)
    method_numbers = {name: index + 1 for index, name in enumerate(selected_methods)}
    if args.phase in ("prepare", "attack") and "cra" not in selected_methods:
        raise ValueError(
            f"--phase {args.phase} is only meaningful when cra is selected"
        )
    image = Path(args.image).expanduser().resolve()
    lora = Path(args.lora_path).expanduser().resolve()
    # Do not resolve these symlinks: resolving a venv interpreter points back
    # to the base Miniconda binary and silently bypasses the venv packages.
    flux_python = Path(sys.executable)
    sam_python = Path(args.sam_python).expanduser()
    for label, path in (("image", image), ("LoRA", lora), ("SAM Python", sam_python)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    target_class, target_label = resolve_target(args.target, args.target_class)
    foreground_prompt = subject_phrase(args.concept_token, args.subject)
    safe_subject = args.subject.strip().replace(" ", "_")
    run_name = args.run_name or f"{image.stem}_{safe_subject}_{target_label.replace(' ', '_')}"
    run_root = Path(args.output_root).expanduser().resolve() / run_name
    if args.phase != "attack" and run_root.exists():
        raise FileExistsError(
            f"Output already exists: {run_root}. Use a new --run-name to avoid overwriting."
        )
    if args.phase == "attack" and not run_root.is_dir():
        raise FileNotFoundError(f"Prepared run does not exist: {run_root}")
    assets = run_root / "assets"
    logs = run_root / "logs"
    cra_root = run_root / "cra"
    jia_root = run_root / "jia"
    cira_root = run_root / "cira"
    cira_preparation_root = cira_root / "preparation"
    for directory in (
        assets,
        logs,
        cra_root,
        jia_root,
        cira_root,
        cira_preparation_root,
    ):
        directory.mkdir(parents=True, exist_ok=True)
    source = assets / f"source{image.suffix.lower()}"
    if args.phase != "attack":
        shutil.copy2(image, source)
    elif not source.is_file():
        matches = sorted(assets.glob("source.*"))
        if len(matches) != 1:
            raise FileNotFoundError(
                f"Expected exactly one prepared source in {assets}, found {len(matches)}"
            )
        source = matches[0]

    visible_anchor = (args.visible_anchor or target_label).strip()
    if not visible_anchor:
        raise ValueError("--visible-anchor cannot be empty")
    anchor_mode = visible_anchor.lower() != target_label.lower()
    default_background_prompt = (
        f"exactly one {visible_anchor} in the distant background, "
        f"a single {visible_anchor}, slightly blurred by shallow depth of field, "
        f"with visible edges, no duplicate animals"
    )
    background_prompt = args.background_prompt or default_background_prompt
    default_teacher_prompt = (
        "Edit only the masked background. Preserve the existing foreground "
        "subject unchanged. In the distant background, exactly one "
        f"{visible_anchor}, slightly blurred by shallow depth of field, with "
        "visible edges. Do not add or modify any foreground subject."
    )
    teacher_prompt = args.teacher_prompt or default_teacher_prompt
    # Separate foreground and background into two clauses so the blur modifier
    # cannot be interpreted as applying to the protected foreground subject.
    default_attack_prompt = (
        f"a realistic photograph with a sharp, in-focus {foreground_prompt} "
        f"in the foreground; in the distant background, exactly one "
        f"{visible_anchor} is slightly blurred by shallow depth of field, "
        f"with visible edges, with no other {visible_anchor} and no duplicate animals"
    )
    attack_prompt = args.attack_prompt or default_attack_prompt
    manifest = {
        "image": str(image),
        "copied_source": str(source),
        "subject": args.subject,
        "source_case": args.source_case,
        "concept_token": args.concept_token,
        "foreground_prompt": foreground_prompt,
        "source_classes": args.source_classes,
        "target_class": target_class,
        "target_label": target_label,
        "visible_anchor": visible_anchor,
        "anchor_mode": anchor_mode,
        "background_prompt": background_prompt,
        "attack_prompt": attack_prompt,
        "teacher_prompt": teacher_prompt,
        "prompt_overrides": {
            "background": args.background_prompt is not None,
            "teacher": args.teacher_prompt is not None,
            "attack": args.attack_prompt is not None,
        },
        "selected_methods": list(selected_methods),
        "lora_path": str(lora),
        "lora_scale": args.lora_scale,
        "seed": args.seed,
        "attack_resolution": args.attack_resolution,
        "composite_resolution": args.composite_resolution,
        "cra": {"classifier_scale": args.attack_scale, "progress": "50->35->50", "attack_steps": 15},
        "jia": {
            "script": str(TEACHER_ORIGINAL),
            "classifier_scale": args.attack_scale,
            "gradient_normalization": "rms",
            "attack_weight": "constant",
            "attack_step_indices": [35, 49],
            "attack_everywhere": True,
            "save_clean_baseline": True,
            "seed": 0,
        },
        "cira": {
            "input": "independently regenerated from original + background_mask + teacher_prompt",
            "preparation_output": "cira/preparation/clean.png",
            "classifier_scale": args.attack_scale,
            "progress": "50->35->50",
            "attack_steps": 15,
            "attack_scope": "global",
        },
    }
    manifest_path = run_root / "manifest.json"
    if args.phase == "attack":
        prepared_manifest = read_json(manifest_path)
        if not prepared_manifest:
            raise FileNotFoundError(f"Prepared manifest missing: {manifest_path}")
        for key, expected in (
            ("image", str(image)),
            ("subject", args.subject),
            ("target_class", target_class),
            ("lora_path", str(lora)),
        ):
            if prepared_manifest.get(key) != expected:
                raise ValueError(
                    f"Prepared manifest mismatch for {key}: "
                    f"{prepared_manifest.get(key)!r} != {expected!r}"
                )
        qwen_gate = read_json(run_root / "qwen_gate.json")
        classifier_gate = read_json(run_root / "classifier_gate_normal.json")
        gate_passed = (
            (qwen_gate and qwen_gate.get("pass") is True)
            or (qwen_gate and qwen_gate.get("forced_best") is True)
            or (classifier_gate and classifier_gate.get("pass") is True
                and 1 <= int(classifier_gate.get("target_rank", -1)) <= 10)
        )
        if not args.dry_run and not gate_passed:
            raise RuntimeError(
                f"Neither Qwen nor classifier Top-10 gate has passed for {run_root}"
            )
        manifest = prepared_manifest
        visible_anchor = str(manifest.get("visible_anchor", target_label))
        anchor_mode = bool(manifest.get("anchor_mode", False))
        background_prompt = manifest["background_prompt"]
        prepared_visible_anchor = str(manifest.get("visible_anchor", target_label))
        qwen_target_phrase = str(manifest.get("qwen_target_phrase", prepared_visible_anchor))
        # Only the composite-return route follows the Qwen-approved target phrase.
        # Teacher inpainting and its clean-inpainting return remain independent.
        attack_prompt = re.sub(
            rf"\b{re.escape(prepared_visible_anchor)}\b",
            qwen_target_phrase,
            manifest["attack_prompt"],
            flags=re.IGNORECASE,
        )
        # Inpainting methods use the current background-only prompt and never
        # inherit the composite/Qwen wording from a prepared manifest.
        teacher_prompt = re.sub(
            rf"\b{re.escape(prepared_visible_anchor)}\b",
            qwen_target_phrase,
            teacher_prompt,
            flags=re.IGNORECASE,
        )
        manifest["teacher_prompt"] = teacher_prompt
        manifest["effective_visible_anchor"] = qwen_target_phrase
        manifest["selected_methods"] = list(selected_methods)
        # Use the exact seed that produced the Qwen-approved composite.
        args.seed = int(manifest["seed"])
        manifest["effective_cra_prompt"] = attack_prompt
        manifest["teacher_prompt_qwen_adjusted"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest["phase"] = "prepared" if args.phase == "prepare" else "full"
        manifest["default_background_prompt"] = default_background_prompt
        manifest["default_teacher_prompt"] = default_teacher_prompt
        manifest["default_attack_prompt"] = default_attack_prompt
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    started = time.time()

    # Gate 1: segmentation. Merge multiple same-concept instances into one mask.
    if args.phase != "attack":
        sam_command = [
            str(sam_python), str(HERE / "sam3_single.py"),
            "--image", str(source), "--prompt", args.subject,
            "--output-dir", str(assets),
        ]
        code = run_logged("SAM3 subject-mask union", sam_command, logs / "01_sam3.log", env, args.dry_run)
        if code != 0:
            error = read_json(assets / "sam3_error.json") or {"exit_code": code}
            (run_root / "automation_error.json").write_text(
                json.dumps({"stage": "sam3", **error}, indent=2), encoding="utf-8"
            )
            raise SystemExit(code)

    composite = assets / "composite.png"
    composite_command = [
        str(flux_python), str(HERE / "generate_composite.py"),
        "--source_image", str(source),
        "--subject_mask", str(assets / "subject_mask.png"),
        "--prompt", background_prompt,
        "--lora_path", str(lora), "--lora_scale", str(args.lora_scale),
        "--num_steps", "30", "--guidance_scale", "3.5",
        "--seed", str(args.seed),
        "--height", str(args.composite_resolution),
        "--width", str(args.composite_resolution),
        "--mask_feather", "4", "--output", str(composite),
        "--local_files_only",
    ]
    if args.phase != "attack" and "cra" in selected_methods:
        code = run_logged("FLUX composite", composite_command, logs / "02_composite.log", env, args.dry_run)
        if code != 0:
            raise SystemExit(code)
        source_gate_record = {
            "pass": True,
            "bypassed": True,
            "decision_source": "source_gate_nonblocking_by_design",
            "source_case": args.source_case,
            "classifier_image": str(composite),
        }
        (run_root / "source_gate.json").write_text(
            json.dumps(source_gate_record, indent=2), encoding="utf-8"
        )
        manifest["source_gate"] = source_gate_record
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        if args.phase == "prepare":
            manifest["phase"] = "awaiting_qwen_gate"
            manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
            print("\nPREPARATION COMPLETE | AWAITING QWEN GATE")
            print(f"QWEN INPUT | {composite.with_name(composite.stem + '_GENERATED.png')}")
            return

    stages: list[dict] = []

    if "cra" in selected_methods:
        # Preserve the original composite/Qwen prompt and gate behavior.
        return_command = [
            str(flux_python), str(HERE / "run_cra.py"),
            "--base-flux-attack", str(HERE / "cra_attack.py"),
            "--full-steps", "50", "--return-to-progress", "35",
            "--run-name", f"scale{args.attack_scale:g}_late15", "--output-root", str(cra_root),
            "--input_image", str(composite), "--single_image_attack",
            "--prompt", attack_prompt, "--inversion_prompt", attack_prompt,
            "--target_class", str(target_class), "--target_label", target_label,
            "--source_classes", args.source_classes,
            "--guidance_scale", "2.5", "--inversion_guidance_scale", "2.5",
            "--recon_guidance_scale", "1.0", "--pivot_correction", "1.0",
            "--classifier_scale", str(args.attack_scale), "--normalize_grad",
            "--objective", "ce", "--attack_sign", "1",
            "--height", str(args.attack_resolution), "--width", str(args.attack_resolution),
            "--seed", str(args.seed), "--local_files_only", "--skip_clean_baseline",
            "--lora_path", str(lora), "--lora_scale", str(args.lora_scale),
        ]
        return_dir = cra_root / f"scale{args.attack_scale:g}_late15"
        code = run_logged(
            "CRA | carrier finite return 50->35->50 with global guidance",
            return_command, logs / "05_return.log", env, args.dry_run,
            method_number=method_numbers["cra"], method_total=method_total,
            method_output=return_dir,
        )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(
                flux_python, return_dir, assets / "subject_mask.png",
                logs / "05b_return_gradcam.log", env, args.dry_run,
            )
        stages.append({"method": "cra", "exit_code": code, "output": str(return_dir), "gradcam": gradcam})

    if ({"jia", "cira"} & set(selected_methods)) and not TEACHER_ORIGINAL.is_file():
        raise FileNotFoundError(f"Teacher original script not found: {TEACHER_ORIGINAL}")

    if "jia" in selected_methods:
        teacher_dir = jia_root / f"aligned_rms{args.attack_scale:g}_late35_49"
        gate_code = run_logged(
            "TEACHER CLEAN INPAINTING GATE | generate clean, then classifier/Qwen",
            [str(flux_python), str(INPAINTING_CLEAN_GATE),
             "--run-root", str(run_root), "--output-dir", str(teacher_dir),
             "--route", "jia"],
            logs / "06a_teacher_clean_gate.log", env, args.dry_run,
        )
        if gate_code != 0:
            raise RuntimeError(f"Teacher clean inpainting gate failed with exit code {gate_code}")
        teacher_gate = {} if args.dry_run else read_json(teacher_dir / "clean_gate.json")
        approved_teacher_prompt = str(teacher_gate.get("prompt", teacher_prompt))
        approved_teacher_seed = int(teacher_gate.get("seed", args.seed))
        teacher_command = [
            str(flux_python), str(TEACHER_ORIGINAL),
            "--image", str(source), "--mask", str(assets / "background_mask.png"),
            "--inpaint-prompt", approved_teacher_prompt, "--target-class", str(target_class),
            "--attack-scale", str(args.attack_scale), "--attack-grad-norm", "rms", "--attack-weight", "constant",
            "--flux-steps", "50", "--attack-start-step", "35", "--attack-end-step", "49",
            "--seed", str(approved_teacher_seed), "--output-dir", str(teacher_dir),
            "--no-cpu-offload", "--attack-everywhere",
        ]
        code = run_logged(
            "JOINT INPAINTING | joint generation and adversarial optimization",
            teacher_command, logs / "06_jia.log", env, args.dry_run,
            method_number=method_numbers["jia"], method_total=method_total,
            method_output=teacher_dir,
        )
        if code == 0:
            if not args.dry_run:
                extract_teacher_step_probabilities(logs / "06_jia.log", target_label, teacher_dir / "attack_steps.csv")
            code = run_logged(
                "Teacher original matched final evaluation",
                [str(flux_python), str(HERE / "evaluate_imagenet.py"),
                 "--image", str(teacher_dir / "attacked.png"), "--target-class", str(target_class),
                 "--output", str(teacher_dir / "attack_eval.json")],
                logs / "07_teacher_eval.log", env, args.dry_run,
            )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(flux_python, teacher_dir, assets / "subject_mask.png", logs / "07b_teacher_gradcam.log", env, args.dry_run)
        stages.append({"method": "jia", "exit_code": code, "output": str(teacher_dir), "gradcam": gradcam})

    if "cira" in selected_methods:
        # Independently regenerate clean inpainting from the original image,
        # background mask, and inpainting prompt. Never reuse method 2 output.
        preparation_dir = cira_preparation_root / f"aligned_rms{args.attack_scale:g}_late35_49"
        code = run_logged(
            "INPAINTING RETURN CLEAN GATE | independently generate clean, then classifier/Qwen",
            [str(flux_python), str(INPAINTING_CLEAN_GATE),
             "--run-root", str(run_root), "--output-dir", str(preparation_dir),
             "--route", "inpainting_return"],
            logs / "08_inpainting_return_preparation.log", env, args.dry_run,
        )
        clean_inpainting = preparation_dir / "clean.png"
        if code != 0:
            raise RuntimeError(f"Independent inpainting-return preparation failed with exit code {code}")
        if not args.dry_run and not clean_inpainting.is_file():
            raise FileNotFoundError(f"Independent clean inpainting was not created: {clean_inpainting}")
        preparation_gate = {} if args.dry_run else read_json(preparation_dir / "clean_gate.json")
        approved_return_prompt = str(preparation_gate.get("prompt", teacher_prompt))
        approved_return_seed = int(preparation_gate.get("seed", args.seed))

        inpainting_return_dir = cira_root / f"scale{args.attack_scale:g}_late15"
        inpainting_return_command = [
            str(flux_python), str(HERE / "run_cra.py"),
            "--run-name", f"scale{args.attack_scale:g}_late15", "--output-root", str(cira_root),
            "--base-flux-attack", str(HERE / "cra_attack.py"),
            "--full-steps", "50", "--return-to-progress", "35",
            "--input_image", str(clean_inpainting), "--single_image_attack",
            "--prompt", approved_return_prompt, "--inversion_prompt", approved_return_prompt,
            "--target_class", str(target_class), "--target_label", target_label,
            "--source_classes", args.source_classes,
            "--guidance_scale", "2.5", "--inversion_guidance_scale", "2.5",
            "--recon_guidance_scale", "1.0", "--pivot_correction", "1.0",
            "--classifier_scale", str(args.attack_scale), "--normalize_grad", "--objective", "ce", "--attack_sign", "1",
            "--height", str(args.attack_resolution), "--width", str(args.attack_resolution),
            "--seed", str(approved_return_seed), "--local_files_only", "--skip_clean_baseline",
            "--lora_path", str(lora), "--lora_scale", str(args.lora_scale),
        ]
        code = run_logged(
            "CIRA | clean inpainting followed by 15-step global finite return",
            inpainting_return_command, logs / "09_cira.log", env, args.dry_run,
            method_number=method_numbers["cira"], method_total=method_total,
            method_output=inpainting_return_dir,
        )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(flux_python, inpainting_return_dir, assets / "subject_mask.png", logs / "09b_inpainting_return_gradcam.log", env, args.dry_run)
        stages.append({"method": "cira", "exit_code": code, "output": str(inpainting_return_dir), "gradcam": gradcam})

    # ResNet-50 remains the only white-box attack model. Reuse each fixed
    # attacked.png for inference-only evaluation on the black-box models.
    for stage_index, stage in enumerate(stages, start=1):
        output = Path(stage["output"])
        if stage["exit_code"] != 0:
            stage["transfer_exit_code"] = None
            continue
        transfer_code = run_logged(
            f"BLACK-BOX TRANSFER EVALUATION | {stage['method']}",
            [
                str(flux_python), str(HERE / "evaluate_transferability.py"),
                "--image", str(output / "attacked.png"),
                "--target-class", str(target_class),
                "--models", *args.transfer_models,
                "--output-json", str(output / "transfer_eval.json"),
                "--output-csv", str(output / "transfer_eval.csv"),
            ],
            logs / f"10{stage_index}_{stage['method']}_transfer_eval.log",
            env, args.dry_run,
        )
        stage["transfer_exit_code"] = transfer_code
        if transfer_code != 0:
            raise RuntimeError(
                f"Black-box transfer evaluation failed for {stage['method']} "
                f"with exit code {transfer_code}"
            )

    for stage in stages:
        output = Path(stage["output"])
        stage["final_eval"] = read_json(output / "attack_eval.json")
        stage["transfer_eval"] = read_json(output / "transfer_eval.json")
        stage["steps"] = step_summary(output / "attack_steps.csv")
    write_transfer_summary(stages, run_root / "transferability_summary.csv")
    contact_path = None
    transfer_contact_path = None
    if not args.dry_run:
        contact_path = export_run_contact(
            run_root, run_name, source, assets, stages, selected_methods,
            cira_preparation_root, jia_root, args.attack_scale,
        )
        transfer_contact_path = export_transfer_contact(run_root, run_name, stages)
    summary = {
        "run_root": str(run_root),
        "elapsed_seconds": time.time() - started,
        "target_class": target_class,
        "target_label": target_label,
        "selected_methods": list(selected_methods),
        "contact": None if contact_path is None else str(contact_path),
        "transferability_contact": None if transfer_contact_path is None else str(transfer_contact_path),
        "methods": stages,
    }
    (run_root / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"\n{'#' * 96}")
    print(f"AUTOMATION COMPLETE | {method_total} SELECTED METHOD(S) FINISHED")
    print(f"RUN ROOT | {run_root}")
    print(f"{'#' * 96}")
    for index, stage in enumerate(stages, 1):
        status = "SUCCESS" if stage["exit_code"] == 0 else f"FAILED ({stage['exit_code']})"
        print(f"METHOD {index}/{method_total} | {stage['method']} | {status}")
        print(f"RESULT DIRECTORY | {stage['output']}")
    print(f"{'#' * 96}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
