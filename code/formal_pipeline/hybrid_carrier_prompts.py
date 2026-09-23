"""Build Hybrid Carrier prompts from the target-specific feature catalog."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re

from carrier_catalog import lookup_carrier


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
    forbidden = str(entry["forbidden_clause"]).strip().rstrip(".")
    if _contains_label(forbidden, str(entry["target_label"])):
        forbidden = ""
    core = (
        f"one small fully visible {carrier} in the upper background, {feature}. "
        f"It must {identity}. "
        + (f"{forbidden}. " if forbidden else "") + "No duplicate."
    )
    prompts = {
        "background_prompt": core[0].upper() + core[1:],
        "jia_prompt": (
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


def select_hybrid_spec(target_class: int, target_label: str, subject: str,
                       concept_token: str, feature_clause: str = "",
                       identity_clause: str = "", forbidden_clause: str = "") -> tuple[dict, dict]:
    carrier = lookup_carrier(target_class)
    catalog_entry = load_hybrid_catalog().get(target_class)
    if catalog_entry:
        entry = dict(catalog_entry)
        if int(entry["carrier_class"]) != int(carrier["carrier_class"]):
            raise ValueError("Hybrid and non-target carrier catalogs disagree")
    else:
        if not all((feature_clause, identity_clause)):
            raise SystemExit("This target needs HYBRID_FEATURE_CLAUSE and HYBRID_IDENTITY_CLAUSE")
        entry = {
            "target_class": target_class,
            "target_label": target_label,
            "carrier_class": carrier["carrier_class"],
            "carrier_label": carrier["carrier_label"],
            "feature_clause": feature_clause,
            "identity_clause": identity_clause,
            "forbidden_clause": forbidden_clause,
        }
    foreground = f"{concept_token} {subject}".strip()
    prompts = build_hybrid_prompts(entry, foreground, subject)
    return entry, prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-class", required=True, type=int)
    parser.add_argument("--target-label", required=True)
    parser.add_argument("--subject", required=True)
    parser.add_argument("--concept-token", default="[V]")
    parser.add_argument("--feature-clause", default="")
    parser.add_argument("--identity-clause", default="")
    parser.add_argument("--forbidden-clause", default="")
    args = parser.parse_args()
    entry, prompts = select_hybrid_spec(
        args.target_class, args.target_label, args.subject, args.concept_token,
        args.feature_clause, args.identity_clause, args.forbidden_clause,
    )
    for value in (
        str(entry["carrier_class"]), str(entry["carrier_label"]),
        prompts["background_prompt"], prompts["jia_prompt"], prompts["attack_prompt"],
    ):
        print(value)


if __name__ == "__main__":
    main()
