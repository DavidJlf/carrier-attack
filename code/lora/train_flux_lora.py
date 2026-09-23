#!/usr/bin/env python3
"""Validated launcher for the official Diffusers FLUX DreamBooth LoRA trainer."""

from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path


DEFAULT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
DEFAULT_TRAINER = "train_dreambooth_lora_flux.py"


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Launch a FLUX-native DreamBooth LoRA run for one subject.",
    )
    p.add_argument("--instance_data_dir", required=True)
    p.add_argument("--instance_prompt", required=True)
    p.add_argument("--output_dir", required=True)
    p.add_argument("--model_id", default=DEFAULT_MODEL)
    p.add_argument("--trainer", default=DEFAULT_TRAINER)
    p.add_argument("--accelerate", default="accelerate")
    p.add_argument("--resolution", type=int, default=1024)
    p.add_argument("--max_train_steps", type=int, default=1250)
    p.add_argument("--learning_rate", type=float, default=1e-4)
    p.add_argument("--rank", type=int, default=16)
    p.add_argument("--lora_alpha", type=int, default=16)
    p.add_argument("--checkpointing_steps", type=int, default=250)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--validation_prompt")
    p.add_argument("--resume_from_checkpoint")
    p.add_argument("--train_text_encoder", action="store_true")
    p.add_argument("--execute", action="store_true")
    return p


def main() -> None:
    args = parser().parse_args()
    data = Path(args.instance_data_dir)
    trainer = Path(args.trainer)
    images = sorted(
        p for p in data.iterdir() if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
    ) if data.is_dir() else []
    if not trainer.is_file():
        raise FileNotFoundError(f"Official FLUX trainer not found: {trainer}")
    if not images:
        raise FileNotFoundError(f"No training images found in: {data}")

    cmd = [
        args.accelerate, "launch",
        "--mixed_precision", "bf16",
        str(trainer),
        "--pretrained_model_name_or_path", args.model_id,
        "--instance_data_dir", str(data),
        "--instance_prompt", args.instance_prompt,
        "--output_dir", args.output_dir,
        "--resolution", str(args.resolution),
        "--train_batch_size", "1",
        "--gradient_accumulation_steps", "1",
        "--learning_rate", str(args.learning_rate),
        "--lr_scheduler", "constant",
        "--lr_warmup_steps", "0",
        "--max_train_steps", str(args.max_train_steps),
        "--rank", str(args.rank),
        "--lora_alpha", str(args.lora_alpha),
        "--mixed_precision", "bf16",
        "--gradient_checkpointing",
        "--cache_latents",
        "--use_8bit_adam",
        "--checkpointing_steps", str(args.checkpointing_steps),
        "--checkpoints_total_limit", "3",
        "--seed", str(args.seed),
        "--report_to", "tensorboard",
    ]
    if args.validation_prompt:
        cmd += [
            "--validation_prompt", args.validation_prompt,
            "--num_validation_images", "2",
            "--validation_epochs", "25",
        ]
    if args.resume_from_checkpoint:
        cmd += ["--resume_from_checkpoint", args.resume_from_checkpoint]
    if args.train_text_encoder:
        cmd += ["--train_text_encoder"]
    else:
        # Precomputation releases CLIP/T5 during Transformer-only LoRA training.
        # It cannot be used when the text encoder itself is trainable.
        cmd += ["--pre_compute_text_embeddings"]

    print(f"Found {len(images)} training images")
    print("Command:\n" + shlex.join(cmd))
    if not args.execute:
        print("Dry run only. Add --execute to start training.")
        return
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
