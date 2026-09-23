#!/usr/bin/env python3
"""Batch-check composite_GENERATED.png files with one resident Qwen-VL model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
from typing import Any

import torch
from PIL import Image
from qwen_vl_utils import process_vision_info
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from torchvision.models import ResNet50_Weights, resnet50


DEFAULT_MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"
REQUIRED_PROMPT_PHRASES = (
    "exactly one",
    "in the distant background",
    "with visible edges",
    "no duplicate animals",
)
ALLOWED_ACTIONS = {
    "accept", "rewrite_prompt", "change_seed", "rewrite_prompt_and_change_seed"
}

TARGET_RUBRICS = {
    "backpack": "A manufactured wearable bag with an unmistakable bag body and visible shoulder straps, handle, zipper, pockets, or seams. An animal, furry body, animal-bag hybrid, or vague lump is NOT a backpack.",
    "chihuahua": "A recognizable small Chihuahua dog with canine face, four-legged dog anatomy, and characteristic large upright ears. A generic cub, fox, cat, or malformed animal is NOT enough.",
    "laptop": "A portable computer with a clearly recognizable screen/display and keyboard base, normally connected by a hinge. A bag, box, tablet alone, or malformed slab is NOT a laptop.",
    "teddy": "A stuffed teddy bear plush toy made of fabric/fur with toy proportions and visible toy cues such as stitching or seams. A living bear, lion cub, other live animal, or realistic cub is NOT a teddy.",
    "police car": "A recognizable police motor vehicle with a coherent car body plus clear police markings and/or roof emergency lights. An ordinary car, toy-like blob, or malformed vehicle is NOT enough.",
    "police van": "A recognizable police van with a coherent van body and clear police identity such as a roof emergency light bar, POLICE marking, badge, or police livery. A sedan, ordinary van, ambulance, or malformed vehicle is not the target class.",
}
CANONICAL_TARGET_PHRASES = {
    "backpack": "clearly manufactured backpack with shoulder straps, handle, zippers, pockets, and fabric seams",
    "chihuahua": "recognizable small Chihuahua dog with large upright ears and coherent canine anatomy",
    "laptop": "open laptop computer with a clearly visible screen, keyboard base, and hinge",
    "teddy": "teddy bear plush toy made of soft fabric with stitched facial features and visible seams",
    "police car": "recognizable marked police car with coherent vehicle body, police markings, and roof emergency lights",
    "police van": "recognizable police van with coherent van body and clear roof emergency light bar or police markings",
}


def rebuild_fixed_prompt(target_phrase: str, current_prompt: str) -> str:
    blur_phrase = (
        "heavily blurred by shallow depth of field"
        if "heavily blurred by shallow depth of field" in current_prompt.lower()
        else "slightly blurred by shallow depth of field"
    )
    return (
        f"exactly one {target_phrase} in the distant background, "
        f"a single {target_phrase}, {blur_phrase}, "
        "with visible edges, no duplicate animals"
    )


SYSTEM_PROMPT = """You are a strict visual quality-control agent for a controlled
image-generation experiment. Inspect only the supplied composite_GENERATED.png.
This is the raw generated background before a foreground source is pasted over it.
Future partial or complete occlusion by the foreground is acceptable and irrelevant.
Do not evaluate a merged composite, inpainting result, or attacked image.

Your primary and highest-priority task is TARGET-CLASS PRESENCE: decide whether
the image contains one recognizable instance of the intended original semantic
target class. Do not invent narrower subtypes beyond the named target class and
do not demand every typical attribute. Minor blur, missing small details, ordinary
viewpoint variation, and modest cosmetic imperfections are acceptable. Fail for
wrong class, missing target, multiple targets, or deformation so severe that the
target class is no longer recognizable. This is not an aesthetic-quality contest.

The target name is only a question, never evidence. Do not infer that the target is
present from the prompt or from one suggestive fragment. First identify the visible
object independently from pixels. Pass only when exactly one visually unambiguous,
coherent instance of the intended ImageNet target is truly present, has the correct
visual subtype, is recognizable before merging, and has no severe anatomical or
structural defect. Vague resemblance, pareidolia, a target-animal hybrid, a malformed
blob, or an object that requires imagination MUST fail. A real bear or cub is not a
teddy bear plush toy. A furry animal with a handle-like shape is not a backpack.

An ImageNet ResNet-50 result is supplied as auxiliary evidence. It is not the sole
judge, but strong semantic contradiction must be taken seriously. If the classifier
strongly identifies an incompatible object and ranks the target very poorly, PASS is
forbidden unless the pixels show unmistakable target-defining parts. When uncertain,
fail safely and improve the prompt; never award PASS merely to keep the pipeline moving.

If invalid, preserve the current prompt and make the smallest possible revision.
Keep its structure, blur wording, visible-edge wording, and duplicate constraint.
Clarify only the target noun phrase when the subtype is wrong. If the prompt is
already correct but generation is malformed, keep the prompt unchanged and request
a seed change. Do not add a scene, source object, style, camera, lighting, overlap,
occlusion, or foreground-separation constraint. Return JSON only."""

INPAINT_SYSTEM_PROMPT = """You are a strict visual quality-control advisor for a
masked background-inpainting experiment. Inspect the supplied clean inpainting.
The original foreground subject must remain the only foreground subject. The image
must contain exactly one recognizable instance of the intended target in the
background. Fail if the target is missing, malformed, duplicated, or if generation
adds a second foreground subject or duplicates the original subject. The classifier
has already failed to place the target in Top-10, so you may not pass this image.
Recommend the smallest target-phrase clarification or a seed change. Never request
another foreground subject. Return JSON only."""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-roots", nargs="+", required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--attempt", type=int, required=True)
    parser.add_argument("--image-name", default="composite_GENERATED.png")
    parser.add_argument("--image-path", default="")
    parser.add_argument("--classifier-image-path", default="")
    parser.add_argument("--prompt-key", default="")
    parser.add_argument("--inpainting-result", action="store_true")
    parser.add_argument("--record-suffix", default="")
    parser.add_argument("--classifier-failed-advisor", action="store_true")
    parser.add_argument("--skip-classifier", action="store_true")
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def extract_json(text: str) -> dict[str, Any]:
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.I | re.S)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise ValueError(f"Qwen did not return JSON: {text!r}")
        value = json.loads(cleaned[start:end + 1])
    if not isinstance(value, dict):
        raise ValueError("Qwen result must be a JSON object")
    return value


def validate_result(result: dict[str, Any], current_prompt: str, target: str) -> dict[str, Any]:
    passed = result.get("pass") is True
    action = result.get("recommended_action")
    adjusted = str(result.get("adjusted_prompt") or "").strip()
    if action not in ALLOWED_ACTIONS:
        raise ValueError(f"Invalid recommended_action: {action!r}")
    if passed:
        result["recommended_action"] = "accept"
        result["adjusted_prompt"] = current_prompt
        return result
    failures = set(result.get("failure_codes") or [])
    if "WRONG_SUBTYPE" in failures and action != "rewrite_prompt":
        raise ValueError(
            "WRONG_SUBTYPE requires a prompt-only rewrite; changing the seed at the same time is forbidden"
        )
    if action == "accept":
        raise ValueError("Failed result cannot recommend accept")
    if action == "change_seed":
        result["adjusted_prompt"] = current_prompt
        return result
    target_phrase = str(result.get("adjusted_target_phrase") or "").strip()
    if action in {"rewrite_prompt", "rewrite_prompt_and_change_seed"} and target.lower() not in target_phrase.lower():
        target_phrase = CANONICAL_TARGET_PHRASES.get(target.lower(), target)
        result["adjusted_target_phrase"] = target_phrase
        adjusted = rebuild_fixed_prompt(target_phrase, current_prompt)
        result["adjusted_prompt"] = adjusted
    if action in {"rewrite_prompt", "rewrite_prompt_and_change_seed"} and target_phrase:
        target_phrase = re.sub(r"^(?:a|an|the)\s+", "", target_phrase, flags=re.IGNORECASE)
        result["adjusted_target_phrase"] = target_phrase
        adjusted = rebuild_fixed_prompt(target_phrase, current_prompt)
        result["adjusted_prompt"] = adjusted
    if adjusted == current_prompt and target_phrase and target_phrase.lower() != target.lower():
        adjusted = rebuild_fixed_prompt(target_phrase, current_prompt)
        result["adjusted_prompt"] = adjusted
    if not adjusted:
        raise ValueError("Prompt rewrite requested without adjusted_prompt")
    if adjusted == current_prompt:
        raise ValueError("Prompt rewrite requested but adjusted_prompt did not change")
    lowered = adjusted.lower()
    missing = [phrase for phrase in REQUIRED_PROMPT_PHRASES if phrase not in lowered]
    expected_blur = (
        "heavily blurred by shallow depth of field"
        if "heavily blurred by shallow depth of field" in current_prompt.lower()
        else "slightly blurred by shallow depth of field"
    )
    if expected_blur not in lowered:
        missing.append(expected_blur)
    if target.lower() not in lowered:
        missing.append(f"target label {target!r}")
    if missing:
        raise ValueError(f"Adjusted prompt violates fixed template; missing {missing}")
    return result


def classifier_evidence(model, weights, image_path: Path, target_class: int) -> dict[str, Any]:
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    with torch.inference_mode():
        probabilities = model(weights.transforms()(image).unsqueeze(0).to(device))[0].softmax(0)
    target_probability = probabilities[target_class]
    top_prob, top_idx = probabilities.topk(5)
    categories = weights.meta["categories"]
    return {
        "target_class": target_class,
        "target_label": categories[target_class],
        "target_probability": float(target_probability),
        "target_rank": int((probabilities > target_probability).sum().item() + 1),
        "top5": [
            {"class": int(i), "label": categories[int(i)], "probability": float(p)}
            for p, i in zip(top_prob, top_idx)
        ],
    }


def user_prompt(target: str, current_prompt: str, evidence: dict[str, Any], advisor_mode: bool = False) -> str:
    rubric = TARGET_RUBRICS.get(target.lower(), "Require a coherent, unmistakable instance of the exact named target.")
    advisor_instruction = "" if not advisor_mode else """
MODE: CLASSIFIER-FAILED PROMPT ADVISOR.
The primary classifier has already rejected this image because the intended target
was not within Top-10. You may not pass it. Diagnose what was generated and return
a minimal rewrite of CURRENT GENERATION PROMPT. adjusted_target_phrase must
describe the intended target, never the incorrect visible object.
"""
    return f"""INTENDED TARGET:\n{target}\n\nTARGET-SPECIFIC VISUAL STANDARD:\n{rubric}\n\nCURRENT GENERATION PROMPT:\n{current_prompt}
{advisor_instruction}

RESNET-50 AUXILIARY EVIDENCE ON THIS EXACT IMAGE:
{json.dumps(evidence, indent=2)}

If evidence status is "not_run", no classifier was used for this accessory image;
judge target-class presence directly from pixels.

Required inspection order:
1. Describe actual_object from pixels without trusting INTENDED TARGET.
2. Check the target-specific defining parts and all explicit exclusions above.
3. Compare with classifier evidence. If it strongly contradicts the target, inspect
   for unmistakable defining parts; without them, target_present must be false.
4. Fail any malformed hybrid, severe defect, or merely suggestive silhouette.

Inspect the supplied raw generated-background image and return exactly this JSON:
{{
  "pass": true,
  "actual_object": "",
  "target_present": true,
  "correct_visual_identity": true,
  "correct_visual_subtype": true,
  "exactly_one_target": true,
  "recognizable_before_merging": true,
  "severe_anatomical_defect": false,
  "severe_structural_defect": false,
  "failure_codes": [],
  "failure_explanation": "",
  "recommended_action": "accept",
  "adjusted_target_phrase": "",
  "adjusted_prompt": ""
}}

Allowed actions: accept, rewrite_prompt, change_seed, rewrite_prompt_and_change_seed.
Allowed failure codes: TARGET_MISSING, WRONG_OBJECT, WRONG_SUBTYPE,
MULTIPLE_TARGETS, TARGET_NOT_RECOGNIZABLE, ANATOMICAL_DEFECT,
STRUCTURAL_DEFECT, SEED_QUALITY_FAILURE.

When pass is true, preserve CURRENT GENERATION PROMPT exactly. When false, change
as few words as possible and obey every fixed-prompt constraint from the system message.

Mandatory decision rule:
- adjusted_target_phrase and adjusted_prompt must describe the INTENDED TARGET,
  never the incorrect actual_object currently visible. For example, when a furry
  animal was generated instead of a backpack, strengthen the phrase to a clearly
  manufactured backpack with straps, zippers, pockets, and seams; never request
  another furry animal.
- If failure_codes contains WRONG_SUBTYPE, recommended_action MUST be
  rewrite_prompt. Replace each occurrence of the ambiguous target
  noun phrase with a concise, visually explicit phrase for the intended subtype.
  For example, if target "teddy" was generated as a living bear, use a phrase such
  as "teddy bear plush toy made of soft fabric with stitched facial features".
  Keep the seed unchanged. Never recommend change_seed or
  rewrite_prompt_and_change_seed for WRONG_SUBTYPE.
- Use change_seed alone only for random anatomy/structure defects when visual identity
  and subtype are already correct."""


def main() -> None:
    args = parse_args()
    run_roots = [Path(item).expanduser().resolve() for item in args.run_roots]
    model_path = Path(args.model).expanduser()
    if not model_path.is_dir():
        raise FileNotFoundError(f"Qwen model directory missing: {model_path}")

    classifier_weights = None if args.skip_classifier else ResNet50_Weights.DEFAULT
    classifier_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    classifier = None if args.skip_classifier else resnet50(weights=classifier_weights).to(classifier_device).eval()
    model = None
    processor = None

    totals = {"pass": 0, "fail": 0, "error": 0}
    for index, run_root in enumerate(run_roots, 1):
        manifest_path = run_root / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        target = str(
            manifest.get("gate_target_label")
            or manifest.get("qwen_requested_target")
            or manifest["target_label"]
        )
        current_prompt = str(
            manifest[args.prompt_key] if args.prompt_key else (
                manifest.get("heavy_blur_prompt")
                if args.image_name == "composite_HEAVILY_BLURRED_GENERATED.png"
                else manifest["background_prompt"]
            )
        )
        image_path = Path(args.image_path).expanduser().resolve() if args.image_path else run_root / "assets" / args.image_name
        if not image_path.is_file():
            raise FileNotFoundError(image_path)
        target_class = int(manifest.get("gate_target_class", manifest["target_class"]))
        classifier_image_path = Path(args.classifier_image_path).expanduser().resolve() if args.classifier_image_path else image_path
        evidence = (
            {"status": "not_run", "target_class": target_class, "top5": []}
            if args.skip_classifier
            else classifier_evidence(classifier, classifier_weights, classifier_image_path, target_class)
        )
        evidence["classifier_image"] = str(classifier_image_path)
        if not args.skip_classifier:
            top1 = evidence["top5"][0]
            print(
                f"CLASSIFIER GATE | target={target} class={target_class} | "
                f"rank={evidence['target_rank']} probability={evidence['target_probability']:.6f} | "
                f"top1={top1['label']} ({top1['probability']:.6f})",
                flush=True,
            )
        # Classifier-first gate: a Top-10 hit is required. Qwen only advises
        # regeneration when the requested target is outside the Top-10.
        if args.classifier_failed_advisor and evidence.get("target_rank", 1001) <= 10:
            record = {
                "attempt": args.attempt,
                "run_root": str(run_root),
                "image": str(image_path),
                "classifier_evidence": evidence,
                "target": target,
                "current_prompt": current_prompt,
                "raw_model_output": "",
                "pass": True,
                "target_present": True,
                "correct_visual_identity": True,
                "recommended_action": "accept",
                "adjusted_target_phrase": "",
                "adjusted_prompt": current_prompt,
                "failure_codes": [],
                "failure_explanation": "",
                "decision_source": "classifier_top10",
                "qwen_invoked": False,
            }
            totals["pass"] += 1
            suffix = f"_{args.record_suffix}" if args.record_suffix else ""
            attempt_path = run_root / f"qwen_gate{suffix}_attempt_{args.attempt}.json"
            attempt_path.write_text(json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8")
            (run_root / f"qwen_gate{suffix}.json").write_text(
                json.dumps(record, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"QWEN SKIPPED | classifier Top-10 accepted target={target}", flush=True)
            continue

        if model is None:
            print(f"Loading Qwen for classifier failure advice: {model_path}", flush=True)
            model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
                str(model_path), torch_dtype=torch.bfloat16, device_map="auto",
                local_files_only=True, attn_implementation="sdpa",
            )
            processor = AutoProcessor.from_pretrained(str(model_path), local_files_only=True)
        messages = [
            {"role": "system", "content": [{"type": "text", "text": INPAINT_SYSTEM_PROMPT if args.inpainting_result else SYSTEM_PROMPT}]},
            {"role": "user", "content": [
                {"type": "image", "image": str(image_path)},
                {"type": "text", "text": user_prompt(target, current_prompt, evidence, args.classifier_failed_advisor)},
            ]},
        ]
        text = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = processor(
            text=[text], images=image_inputs, videos=video_inputs,
            padding=True, return_tensors="pt",
        ).to(model.device)
        with torch.inference_mode():
            output_ids = model.generate(**inputs, max_new_tokens=args.max_new_tokens, do_sample=False)
        trimmed = [out[len(inp):] for inp, out in zip(inputs.input_ids, output_ids)]
        raw = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
        record: dict[str, Any] = {
            "attempt": args.attempt,
            "run_root": str(run_root),
            "image": str(image_path),
            "classifier_evidence": evidence,
            "target": target,
            "current_prompt": current_prompt,
            "raw_model_output": raw,
        }
        try:
            parsed = extract_json(raw)
            advisor_rank = evidence.get("target_rank")
            if args.classifier_failed_advisor and (
                (parsed.get("pass") is True and (advisor_rank is None or advisor_rank > 10))
                or parsed.get("recommended_action") == "change_seed"
            ):
                phrase = CANONICAL_TARGET_PHRASES.get(target.lower(), target)
                parsed.update({
                    "pass": False, "target_present": False,
                    "recommended_action": "rewrite_prompt",
                    "adjusted_target_phrase": phrase,
                    "adjusted_prompt": rebuild_fixed_prompt(phrase, current_prompt),
                    "failure_codes": ["TARGET_NOT_RECOGNIZABLE"],
                    "failure_explanation": "Primary classifier rejected the target; prompt rewrite required.",
                })
            result = validate_result(parsed, current_prompt, target)
            result["decision_source"] = "qwen"
            record.update(result)
            totals["pass" if result["pass"] is True else "fail"] += 1
        except Exception as error:
            record.update({
                "pass": False,
                "recommended_action": "change_seed",
                "adjusted_prompt": current_prompt,
                "failure_codes": ["GATE_OUTPUT_ERROR"],
                "failure_explanation": repr(error),
            })
            totals["error"] += 1
        suffix = f"_{args.record_suffix}" if args.record_suffix else ""
        attempt_path = run_root / f"qwen_gate{suffix}_attempt_{args.attempt}.json"
        attempt_path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        if record["pass"] is True:
            (run_root / f"qwen_gate{suffix}.json").write_text(
                json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8"
            )
        print(
            f"QWEN {index}/{len(run_roots)} | {run_root.name} | "
            f"{'PASS' if record['pass'] else 'FAIL'} | {record['recommended_action']}",
            flush=True,
        )
    print(json.dumps({"checked": len(run_roots), **totals}, indent=2))


if __name__ == "__main__":
    main()
