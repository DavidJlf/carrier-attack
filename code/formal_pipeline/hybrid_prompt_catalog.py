"""Validated prompt construction for the 30-target Hybrid Carrier condition."""

from __future__ import annotations

import json
from pathlib import Path
import re


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "hybrid_carrier_prompt_catalog_30.json"


def load_hybrid_catalog(path: Path = DEFAULT_CATALOG) -> dict[int, dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("entries")
    if not isinstance(entries, list) or not entries:
        raise ValueError(f"Hybrid catalog has no entries: {path}")
    by_class: dict[int, dict] = {}
    for raw in entries:
        entry = dict(raw)
        target_class = int(entry["target_class"])
        carrier_class = int(entry["carrier_class"])
        if target_class in by_class:
            raise ValueError(f"Duplicate hybrid target class: {target_class}")
        if target_class == carrier_class:
            raise ValueError(f"Target/carrier collision: {target_class}")
        for key in ("target_label", "carrier_label", "feature_clause", "identity_clause", "forbidden_clause"):
            if not str(entry.get(key, "")).strip():
                raise ValueError(f"Missing {key} for target class {target_class}")
        by_class[target_class] = entry
    return by_class


def _contains_label(text: str, label: str) -> bool:
    return re.search(rf"(?<![a-z]){re.escape(label.lower())}(?![a-z])", text.lower()) is not None


def build_hybrid_prompts(entry: dict, foreground_prompt: str, subject: str) -> dict:
    carrier = str(entry["carrier_label"]).strip()
    feature = str(entry["feature_clause"]).strip().rstrip(".")
    identity = str(entry["identity_clause"]).strip().rstrip(".")
    # Put the localized target attributes first so both FLUX text encoders see
    # the defining Hybrid condition before any lower-priority exclusions.
    core = (
        f"one small fully visible {carrier} in the upper background, {feature}. "
        f"It must {identity}. No duplicate."
    )
    prompts = {
        "background_prompt": core[0].upper() + core[1:],
        "teacher_prompt": (
            f"Keep foreground {subject} unchanged. Inpaint background: {core}"
        ),
        "attack_prompt": (
            f"Realistic photo: sharp {foreground_prompt} foreground; background contains {core}"
        ),
    }
    target_label = str(entry["target_label"])
    for name, prompt in prompts.items():
        if _contains_label(prompt, target_label):
            raise ValueError(f"{name} directly names target {target_label!r}")
        if carrier.lower() not in prompt.lower():
            raise ValueError(f"{name} does not name carrier label {carrier!r}")
    return prompts


def validate_formal_mapping(targets: tuple[dict, ...], hybrid: dict[int, dict],
                            anchor_lookup) -> None:
    target_classes = {int(item["class"]) for item in targets}
    if set(hybrid) != target_classes:
        missing = sorted(target_classes - set(hybrid))
        extra = sorted(set(hybrid) - target_classes)
        raise ValueError(f"Hybrid catalog/formal target mismatch; missing={missing}, extra={extra}")
    for target in targets:
        entry = hybrid[int(target["class"])]
        anchor = anchor_lookup(int(target["class"]))
        expected = (int(anchor["anchor_class"]), str(anchor["anchor_label"]).lower())
        actual = (int(entry["carrier_class"]), str(entry["carrier_label"]).lower())
        if actual != expected:
            raise ValueError(
                f"Hybrid carrier mismatch for {target['label']}: {actual} != {expected}"
            )
