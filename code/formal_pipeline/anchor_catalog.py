#!/usr/bin/env python3
"""Read and validate the precomputed ImageNet-1K visible-anchor catalog."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = HERE / "imagenet1k_anchor_catalog.json"


def load_catalog(path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    entries = payload.get("entries")
    if payload.get("schema_version") != 1 or not isinstance(entries, list):
        raise ValueError(f"Invalid anchor catalog schema: {path}")
    if len(entries) != 1000:
        raise ValueError(f"Anchor catalog must contain 1000 entries, found {len(entries)}")
    indices = [int(item["target_class"]) for item in entries]
    if indices != list(range(1000)):
        raise ValueError("Anchor catalog target_class values must be exactly 0..999 in order")
    for item in entries:
        target = int(item["target_class"])
        candidates = [int(item["anchor_class"]), *map(int, item["backup_anchor_classes"])]
        if len(candidates) != 3 or len(set(candidates)) != 3:
            raise ValueError(f"Target {target} must have three distinct anchor candidates")
        if target in candidates or any(not 0 <= value < 1000 for value in candidates):
            raise ValueError(f"Invalid anchor candidate for target {target}: {candidates}")
        forbidden = {int(value) for value in item.get("forbidden_near_classes", [])}
        if forbidden.intersection(candidates):
            raise ValueError(f"Target {target} uses forbidden near class: {forbidden.intersection(candidates)}")
    return payload


def lookup_anchor(target_class: int, path: Path = DEFAULT_CATALOG) -> dict[str, Any]:
    if not 0 <= target_class < 1000:
        raise ValueError("target_class must be in 0..999")
    item = load_catalog(path)["entries"][target_class]
    if int(item["target_class"]) != target_class:
        raise ValueError(f"Catalog index mismatch at target {target_class}")
    return item
