#!/usr/bin/env python3
"""Generate deterministic candidate subject samples with a trained FLUX LoRA."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="Generate candidate subject samples from a subject-specific FLUX LoRA.",
    )
    parser.add_argument("--model-id", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--lora-path", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-images", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num-steps", type=int, default=30)
    parser.add_argument("--guidance-scale", type=float, default=3.5)
    parser.add_argument("--lora-scale", type=float, default=1.0)
    parser.add_argument("--allow-download", action="store_true")
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    lora_path = Path(args.lora_path).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    if args.num_images < 1:
        raise ValueError("--num-images must be positive")
    if not lora_path.is_file():
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")
    print(f"Model: {args.model_id}")
    print(f"LoRA: {lora_path}")
    print(f"Output: {output_dir}")
    print(f"Images: {args.num_images}; seeds {args.seed}..{args.seed + args.num_images - 1}")
    if not args.execute:
        print("Dry run only. Add --execute after reviewing the paths and prompt.")
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        local_files_only=not args.allow_download,
    )
    pipe.load_lora_weights(
        str(lora_path.parent), weight_name=lora_path.name, adapter_name="subject_lora"
    )
    pipe.set_adapters(["subject_lora"], adapter_weights=[args.lora_scale])
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    for index in range(args.num_images):
        seed = args.seed + index
        generator = torch.Generator(device="cpu").manual_seed(seed)
        image = pipe(
            prompt=args.prompt,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_steps,
            guidance_scale=args.guidance_scale,
            generator=generator,
            max_sequence_length=256,
        ).images[0]
        image.save(output_dir / f"subject_sample_{index:03d}_seed{seed}.png")
        print(f"[{index + 1:03d}/{args.num_images:03d}] seed={seed}")


if __name__ == "__main__":
    main()
