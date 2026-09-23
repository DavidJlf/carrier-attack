#!/usr/bin/env python3
"""Read candidate non-target carriers for ImageNet target classes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "imagenet1k_carrier_catalog.json"


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("entries")
    if payload.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError(f"Invalid carrier catalog schema: {path}")
    if len(entries) != 1000:
        raise ValueError(f"Carrier catalog must contain 1000 entries, found {len(entries)}")
    indices = [int(item["target_class"]) for item in entries]
    if indices != list(range(1000)):
        raise ValueError("Carrier catalog target_class values must be exactly 0..999 in order")
    for item in entries:
        target = int(item["target_class"])
        candidates = [int(item["carrier_class"]), *map(int, item["backup_carrier_classes"])]
        if len(candidates) != 3 or len(set(candidates)) != 3:
            raise ValueError(f"Target {target} must have three distinct carrier candidates")
        if target in candidates or any(not 0 <= value < 1000 for value in candidates):
            raise ValueError(f"Invalid carrier candidate for target {target}: {candidates}")
        forbidden = {int(value) for value in item.get("forbidden_near_classes", [])}
        if forbidden.intersection(candidates):
            raise ValueError(f"Target {target} uses forbidden near class: {forbidden.intersection(candidates)}")
    return payload


def lookup_carrier(target_class: int, path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    if not 0 <= target_class < 1000:
        raise ValueError("target_class must be in 0..999")
    item = load_catalog(path)["entries"][target_class]
    if int(item["target_class"]) != target_class:
        raise ValueError(f"Catalog index mismatch at target {target_class}")
    return item


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Select the non-target carrier for an ImageNet target class.")
    parser.add_argument("--target-class", required=True, type=int)
    args = parser.parse_args()
    selected = lookup_carrier(args.target_class)
    print(f"{selected['carrier_class']} {selected['carrier_label']}")
