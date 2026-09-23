#!/usr/bin/env python3
"""Automate SAM3, carrier construction, and CRA/JIA/CIRA.

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

from carrier_catalog import lookup_carrier
from hybrid_carrier_prompts import select_hybrid_spec


HERE = Path(__file__).resolve().parent
JIA_ATTACK = HERE / "jia_inpainting_attack.py"
INPAINTING_CLEAN_GATE = HERE / "run_inpainting_clean_gate.py"
DEFAULT_OUTPUT_ROOT = Path("outputs")
DEFAULT_SAM_PYTHON = Path(os.environ.get("SAM_PYTHON", sys.executable))
def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-image carrier construction and CRA/JIA/CIRA automation."
    )
    parser.add_argument("--image", required=True, help="Input subject image.")
    parser.add_argument("--target", required=True, help="Exact ImageNet target label.")
    parser.add_argument(
        "--construction-condition", required=True,
        choices=("target_carrier", "non_target_carrier", "hybrid_carrier"),
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
    parser.add_argument("--hybrid-feature-clause", default="")
    parser.add_argument("--hybrid-identity-clause", default="")
    parser.add_argument("--hybrid-forbidden-clause", default="")
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
    cira_preparation_root: Path,
    jia_root: Path,
    attack_scale: float,
) -> Path:
    """Export the white-box results and Grad-CAM contact image."""
    contacts = run_root / "contacts"
    contacts.mkdir(parents=True, exist_ok=True)
    tile, cap, gap, title_h = 390, 120, 18, 64
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
    independent_clean = cira_preparation_root / f"aligned_rms{attack_scale:g}_late35_49" / "clean.png"
    jia_clean = jia_root / f"aligned_rms{attack_scale:g}_late35_49" / "clean.png"
    if "cra" in selected_methods and generated.is_file():
        context_path, context_label = generated, "Composite generated background"
    elif "cira" in selected_methods and independent_clean.is_file():
        context_path, context_label = independent_clean, "Independent masked clean inpainting"
    elif jia_clean.is_file():
        context_path, context_label = jia_clean, "JIA clean baseline"

    cells = [(0, 0, source, "Original source", None, None)]
    if context_path is not None:
        cells.append((1, 0, context_path, context_label, None, None))
    for col, stage in enumerate(stages, 1):
        method_dir = Path(stage["output"])
        evaluation = stage.get("final_eval")
        cam = stage.get("gradcam")
        cells.append((0, col, method_dir / "attacked.png", stage["method"], evaluation, None))
        cells.append((1, col, method_dir / "gradcam_top1/gradcam_overlay.png", "Predicted Top-1 Grad-CAM", evaluation, cam))

    for row, col, path, label, evaluation, cam in cells:
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
        if cam is not None:
            draw.text((x, y + tile + 32), f"Top-1: {cam['explained_label']} P={100*cam['explained_probability']:.2f}%", font=contact_font(16), fill=(20, 55, 120))
            fg, bg = cam.get("cam_energy_foreground_fraction"), cam.get("cam_energy_background_fraction")
            if fg is not None and bg is not None:
                draw.text((x, y + tile + 57), f"CAM FG={100*fg:.1f}% BG={100*bg:.1f}%", font=contact_font(16), fill=(120, 45, 20))

    output = contacts / f"{run_name}_contact.jpg"
    canvas.save(output, quality=94, subsampling=0)
    print(f"CONTACT EXPORTED | {output}", flush=True)
    return output




def extract_jia_step_probabilities(log_path: Path, target_label: str, output: Path) -> None:
    """Extract the jia script's printed target probabilities without changing it."""
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
                 "source": "jia_original_top3_log_rounded"}
            )
    if rows:
        with output.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)


def main() -> None:
    args = parse_args()
    selected_methods = ("cra", "jia", "cira")
    method_total = len(selected_methods)
    method_numbers = {name: index + 1 for index, name in enumerate(selected_methods)}
    image = Path(args.image).expanduser().resolve()
    lora = Path(args.lora_path).expanduser().resolve()
    # Do not resolve these symlinks: resolving a venv interpreter points back
    # to the base Miniconda binary and silently bypasses the venv packages.
    flux_python = Path(sys.executable)
    sam_python = Path(shutil.which(args.sam_python) or args.sam_python).expanduser()
    for label, path in (("image", image), ("LoRA", lora), ("SAM Python", sam_python)):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")

    target_class, target_label = resolve_target(args.target, args.target_class)
    foreground_prompt = subject_phrase(args.concept_token, args.subject)
    hybrid_prompts = None
    if args.construction_condition == "target_carrier":
        visible_carrier, carrier_class = target_label, target_class
    else:
        selected_carrier = lookup_carrier(target_class)
        visible_carrier = str(selected_carrier["carrier_label"])
        carrier_class = int(selected_carrier["carrier_class"])
        if args.construction_condition == "hybrid_carrier":
            hybrid_entry, hybrid_prompts = select_hybrid_spec(
                target_class, target_label, args.subject, args.concept_token,
                args.hybrid_feature_clause, args.hybrid_identity_clause,
                args.hybrid_forbidden_clause,
            )
            if int(hybrid_entry["carrier_class"]) != carrier_class:
                raise ValueError("Hybrid Carrier class differs from the Non-Target Carrier class")
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
    # protected_root = run_root / "protected_mask"
    cra_root = run_root / "cra"
    jia_root = run_root / "jia"
    cira_root = run_root / "cira"
    cira_preparation_root = cira_root / "preparation"
    # inpainting_protected_root = run_root / "inpainting_protected_mask"
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

    carrier_mode = visible_carrier.lower() != target_label.lower()
    default_background_prompt = (
        f"exactly one {visible_carrier} in the distant background, "
        f"a single {visible_carrier}, slightly blurred by shallow depth of field, "
        f"with visible edges, no duplicate animals"
    )
    background_prompt = hybrid_prompts["background_prompt"] if hybrid_prompts else default_background_prompt
    default_jia_prompt = (
        "Edit only the masked background. Preserve the existing foreground "
        "subject unchanged. In the distant background, exactly one "
        f"{visible_carrier}, slightly blurred by shallow depth of field, with "
        "visible edges. Do not add or modify any foreground subject."
    )
    jia_prompt = hybrid_prompts["jia_prompt"] if hybrid_prompts else default_jia_prompt
    # Keep the foreground and background clauses separate.
    default_attack_prompt = (
        f"a realistic photograph with a sharp, in-focus {foreground_prompt} "
        f"in the foreground; in the distant background, exactly one "
        f"{visible_carrier} is slightly blurred by shallow depth of field, "
        f"with visible edges, with no other {visible_carrier} and no duplicate animals"
    )
    attack_prompt = hybrid_prompts["attack_prompt"] if hybrid_prompts else default_attack_prompt
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
        "construction_condition": args.construction_condition,
        "carrier_class": carrier_class,
        "visible_carrier": visible_carrier,
        "carrier_mode": carrier_mode,
        "background_prompt": background_prompt,
        "attack_prompt": attack_prompt,
        "jia_prompt": jia_prompt,
        "selected_methods": list(selected_methods),
        "lora_path": str(lora),
        "lora_scale": args.lora_scale,
        "seed": args.seed,
        "attack_resolution": args.attack_resolution,
        "composite_resolution": args.composite_resolution,
        "cra": {"classifier_scale": args.attack_scale, "progress": "50->35->50", "attack_steps": 15},
        "jia": {
            "script": str(JIA_ATTACK),
            "classifier_scale": args.attack_scale,
            "gradient_normalization": "rms",
            "attack_weight": "constant",
            "attack_step_indices": [35, 49],
            "attack_everywhere": True,
            "save_clean_baseline": True,
            "seed": 0,
        },
        "cira": {
            "input": "independently regenerated from original + background_mask + jia_prompt",
            "preparation_output": "cira/preparation/clean.png",
            "classifier_scale": args.attack_scale,
            "progress": "50->35->50",
            "attack_steps": 15,
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
            ("construction_condition", args.construction_condition),
            ("carrier_class", carrier_class),
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
        visible_carrier = str(manifest.get("visible_carrier", target_label))
        carrier_mode = bool(manifest.get("carrier_mode", False))
        background_prompt = manifest["background_prompt"]
        prepared_visible_carrier = str(manifest.get("visible_carrier", target_label))
        qwen_target_phrase = str(manifest.get("qwen_target_phrase", prepared_visible_carrier))
        # Only the composite-return route follows the Qwen-approved target phrase.
        # JIA and CIRA construction remain independent.
        attack_prompt = re.sub(
            rf"\b{re.escape(prepared_visible_carrier)}\b",
            qwen_target_phrase,
            manifest["attack_prompt"],
            flags=re.IGNORECASE,
        )
        # Inpainting methods use the current background-only prompt and never
        # inherit the composite/Qwen wording from a prepared manifest.
        jia_prompt = re.sub(
            rf"\b{re.escape(prepared_visible_carrier)}\b",
            qwen_target_phrase,
            jia_prompt,
            flags=re.IGNORECASE,
        )
        manifest["jia_prompt"] = jia_prompt
        manifest["effective_visible_carrier"] = qwen_target_phrase
        manifest["selected_methods"] = list(selected_methods)
        # Use the exact seed that produced the Qwen-approved composite.
        args.seed = int(manifest["seed"])
        manifest["effective_cra_prompt"] = attack_prompt
        manifest["jia_prompt_qwen_adjusted"] = False
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    else:
        manifest["phase"] = "prepared" if args.phase == "prepare" else "full"
        manifest["default_background_prompt"] = default_background_prompt
        manifest["default_jia_prompt"] = default_jia_prompt
        manifest["default_attack_prompt"] = default_attack_prompt
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    env = os.environ.copy()
    # Model cache and online/offline behavior follow the caller's environment.
    started = time.time()

    # Gate 1: segmentation. Merge multiple same-concept instances into one mask.
    if args.phase != "attack":
        sam_command = [
            str(sam_python), str(HERE / "sam3_single.py"),
            "--image", str(source), "--prompt", args.subject,
            "--output-dir", str(assets),
            "--merge-multiple",
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
        cra_command = [
            str(flux_python), str(HERE / "run_cra.py"),
            "--base-flux-attack", str(HERE / "cra_flux_attack.py"),
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
        cra_dir = cra_root / f"scale{args.attack_scale:g}_late15"
        code = run_logged(
            "CRA | carrier finite return 50->35->50 with global guidance",
            cra_command, logs / "05_cra.log", env, args.dry_run,
            method_number=method_numbers["cra"], method_total=method_total,
            method_output=cra_dir,
        )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(
                flux_python, cra_dir, assets / "subject_mask.png",
                logs / "05b_cra_gradcam.log", env, args.dry_run,
            )
        stages.append({"method": "cra", "exit_code": code, "output": str(cra_dir), "gradcam": gradcam})

    if ({"jia", "cira"} & set(selected_methods)) and not JIA_ATTACK.is_file():
        raise FileNotFoundError(f"JIA attack script not found: {JIA_ATTACK}")

    if "jia" in selected_methods:
        jia_dir = jia_root / f"aligned_rms{args.attack_scale:g}_late35_49"
        gate_code = run_logged(
            "JIA CLEAN GATE | generate clean, then classifier/Qwen",
            [str(flux_python), str(INPAINTING_CLEAN_GATE),
             "--run-root", str(run_root), "--output-dir", str(jia_dir),
             "--route", "jia"],
            logs / "06a_jia_clean_gate.log", env, args.dry_run,
        )
        if gate_code != 0:
            raise RuntimeError(f"JIA clean inpainting gate failed with exit code {gate_code}")
        jia_gate = {} if args.dry_run else read_json(jia_dir / "clean_gate.json")
        approved_jia_prompt = str(jia_gate.get("prompt", jia_prompt))
        approved_jia_seed = int(jia_gate.get("seed", args.seed))
        jia_command = [
            str(flux_python), str(JIA_ATTACK),
            "--image", str(source), "--mask", str(assets / "background_mask.png"),
            "--inpaint-prompt", approved_jia_prompt, "--target-class", str(target_class),
            "--attack-scale", str(args.attack_scale), "--attack-grad-norm", "rms", "--attack-weight", "constant",
            "--flux-steps", "50", "--attack-start-step", "35", "--attack-end-step", "49",
            "--seed", str(approved_jia_seed), "--output-dir", str(jia_dir),
            "--no-cpu-offload", "--attack-everywhere",
        ]
        code = run_logged(
            "JIA | joint inpainting and adversarial optimization",
            jia_command, logs / "06_jia.log", env, args.dry_run,
            method_number=method_numbers["jia"], method_total=method_total,
            method_output=jia_dir,
        )
        if code == 0:
            if not args.dry_run:
                extract_jia_step_probabilities(logs / "06_jia.log", target_label, jia_dir / "attack_steps.csv")
            code = run_logged(
                "JIA matched final evaluation",
                [str(flux_python), str(HERE / "evaluate_whitebox_resnet50.py"),
                 "--image", str(jia_dir / "attacked.png"), "--target-class", str(target_class),
                 "--output", str(jia_dir / "attack_eval.json")],
                logs / "07_jia_eval.log", env, args.dry_run,
            )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(flux_python, jia_dir, assets / "subject_mask.png", logs / "07b_jia_gradcam.log", env, args.dry_run)
        stages.append({"method": "jia", "exit_code": code, "output": str(jia_dir), "gradcam": gradcam})

    if "cira" in selected_methods:
        # Independently regenerate clean inpainting from the original image,
        # background mask, and inpainting prompt. Never reuse method 2 output.
        preparation_dir = cira_preparation_root / f"aligned_rms{args.attack_scale:g}_late35_49"
        code = run_logged(
            "CIRA CLEAN GATE | independently generate clean, then classifier/Qwen",
            [str(flux_python), str(INPAINTING_CLEAN_GATE),
             "--run-root", str(run_root), "--output-dir", str(preparation_dir),
             "--route", "cira"],
            logs / "08_cira_preparation.log", env, args.dry_run,
        )
        clean_inpainting = preparation_dir / "clean.png"
        if code != 0:
            raise RuntimeError(f"Independent inpainting-return preparation failed with exit code {code}")
        if not args.dry_run and not clean_inpainting.is_file():
            raise FileNotFoundError(f"Independent clean inpainting was not created: {clean_inpainting}")
        preparation_gate = {} if args.dry_run else read_json(preparation_dir / "clean_gate.json")
        approved_cira_prompt = str(preparation_gate.get("prompt", jia_prompt))
        approved_cira_seed = int(preparation_gate.get("seed", args.seed))

        cira_dir = cira_root / f"scale{args.attack_scale:g}_late15"
        cira_command = [
            str(flux_python), str(HERE / "run_cra.py"),
            "--run-name", f"scale{args.attack_scale:g}_late15", "--output-root", str(cira_root),
            "--base-flux-attack", str(HERE / "cra_flux_attack.py"),
            "--full-steps", "50", "--return-to-progress", "35",
            "--input_image", str(clean_inpainting), "--single_image_attack",
            "--prompt", approved_cira_prompt, "--inversion_prompt", approved_cira_prompt,
            "--target_class", str(target_class), "--target_label", target_label,
            "--source_classes", args.source_classes,
            "--guidance_scale", "2.5", "--inversion_guidance_scale", "2.5",
            "--recon_guidance_scale", "1.0", "--pivot_correction", "1.0",
            "--classifier_scale", str(args.attack_scale), "--normalize_grad", "--objective", "ce", "--attack_sign", "1",
            "--height", str(args.attack_resolution), "--width", str(args.attack_resolution),
            "--seed", str(approved_cira_seed), "--local_files_only", "--skip_clean_baseline",
            "--lora_path", str(lora), "--lora_scale", str(args.lora_scale),
        ]
        code = run_logged(
            "CIRA | independent clean inpainting then 15-step global finite return",
            cira_command, logs / "09_cira.log", env, args.dry_run,
            method_number=method_numbers["cira"], method_total=method_total,
            method_output=cira_dir,
        )
        gradcam = None
        if code == 0:
            _, gradcam = run_top1_gradcam(flux_python, cira_dir, assets / "subject_mask.png", logs / "09b_cira_gradcam.log", env, args.dry_run)
        stages.append({"method": "cira", "exit_code": code, "output": str(cira_dir), "gradcam": gradcam})

    for stage in stages:
        output = Path(stage["output"])
        stage["final_eval"] = read_json(output / "attack_eval.json")
        stage["steps"] = step_summary(output / "attack_steps.csv")
    contact_path = None
    if not args.dry_run:
        contact_path = export_run_contact(
            run_root, run_name, source, assets, stages, selected_methods,
            cira_preparation_root, jia_root, args.attack_scale,
        )
    summary = {
        "run_root": str(run_root),
        "elapsed_seconds": time.time() - started,
        "target_class": target_class,
        "target_label": target_label,
        "selected_methods": list(selected_methods),
        "contact": None if contact_path is None else str(contact_path),
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
