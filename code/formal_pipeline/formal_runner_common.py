"""Shared launch helpers extracted from the original three-example runner."""

from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys


HERE = Path(__file__).resolve().parent
PYTHON = Path(os.environ.get("FLUX_PYTHON", sys.executable))
AUTOMATION = HERE / "flux_auomation.py"
ANCHOR_CATALOG = HERE / "imagenet1k_anchor_catalog.json"
ANCHOR_GATE = HERE / "run_anchor_composite_gate.py"
METHODS = "cra,jia,cira"


def slug(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.lower()).strip("_")


def execute(command: list[str], log: Path | None, dry_run: bool, env: dict[str, str]) -> None:
    print("COMMAND |", " ".join(command), flush=True)
    if dry_run:
        return
    if log is None:
        subprocess.run(command, check=True, env=env)
        return
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env
        )
        assert process.stdout is not None
        for line in process.stdout:
            sys.stdout.write(line)
            sys.stdout.flush()
            handle.write(line)
            handle.flush()
        code = process.wait()
    if code:
        raise subprocess.CalledProcessError(code, command)


def automation_command(
    output_root: Path,
    source: dict,
    target: dict,
    variant: str,
    anchor: str | None,
    phase: str,
    attack_scale: float = 0.5,
    prompt_spec: dict | None = None,
) -> list[str]:
    command = [
        str(PYTHON), str(AUTOMATION), "--image", str(source["image"]),
        "--subject", source["subject"], "--source-case", source["source_case"],
        "--target", target["label"], "--target-class", str(target["class"]),
        "--source-classes", source["source_classes"], "--concept-token", "[V]",
        "--lora-path", str(source["lora"]), "--lora-scale", "1.0",
        "--seed", str(source["seed"]), "--methods", METHODS, "--phase", phase,
        "--attack-scale", str(attack_scale), "--output-root", str(output_root),
        "--run-name", f'{source["slug"]}_to_{slug(target["label"])}_{variant}',
    ]
    if anchor:
        command += ["--visible-anchor", anchor]
    if prompt_spec:
        command += [
            "--background-prompt", str(prompt_spec["background_prompt"]),
            "--jia-prompt", str(prompt_spec["teacher_prompt"]),
            "--attack-prompt", str(prompt_spec["attack_prompt"]),
        ]
    return command
