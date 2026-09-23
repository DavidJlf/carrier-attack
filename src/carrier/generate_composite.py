#!/usr/bin/env python3
"""Generate a FLUX-LoRA background and composite the exact masked subject."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
from diffusers import FluxPipeline
from PIL import Image, ImageFilter


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source_image", required=True)
    parser.add_argument("--subject_mask", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--model_id", default="black-forest-labs/FLUX.1-Kontext-dev")
    parser.add_argument("--cache_dir")
    parser.add_argument("--lora_path", required=True)
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--num_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=4.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--mask_feather", type=float, default=2.0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--local_files_only", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_path = Path(args.source_image).expanduser()
    mask_path_in = Path(args.subject_mask).expanduser()
    lora_path = Path(args.lora_path).expanduser()
    for label, path in (
        ("source image", source_path),
        ("subject mask", mask_path_in),
        ("FLUX LoRA", lora_path),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    if args.lora_scale < 0:
        raise ValueError("--lora_scale must be non-negative")

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    source = Image.open(source_path).convert("RGB").resize(
        (args.width, args.height), Image.Resampling.LANCZOS
    )
    mask = Image.open(mask_path_in).convert("L").resize(
        (args.width, args.height), Image.Resampling.NEAREST
    )
    mask = mask.point(lambda value: 255 if value >= 128 else 0)
    if args.mask_feather > 0:
        mask = mask.filter(ImageFilter.GaussianBlur(radius=args.mask_feather))

    print(f"Loading FLUX model: {args.model_id}")
    pipe = FluxPipeline.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir,
        local_files_only=args.local_files_only,
    )
    pipe.load_lora_weights(
        str(lora_path.parent),
        weight_name=lora_path.name,
        adapter_name="concept_lora",
    )
    pipe.set_adapters(["concept_lora"], adapter_weights=[args.lora_scale])
    print(f"Loaded FLUX LoRA: {lora_path} (scale={args.lora_scale})")
    # Kontext + T5 + LoRA can peak near 48 GiB with model-level offload at
    # 1024x1024. Sequential offload moves individual submodules and trades
    # speed for a substantially lower peak, which is safer on the 48 GiB GPU.
    pipe.enable_sequential_cpu_offload()
    pipe.vae.enable_slicing()
    pipe.vae.enable_tiling()

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    generated = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        generator=generator,
        max_sequence_length=256,
    ).images[0].convert("RGB")

    # White mask = exact original subject; black mask = generated background.
    composite = Image.composite(source, generated, mask)
    generated_path = output_path.with_name(output_path.stem + "_GENERATED.png")
    saved_mask_path = output_path.with_name(output_path.stem + "_MASK.png")
    generated.save(generated_path)
    mask.save(saved_mask_path)
    composite.save(output_path)
    print(f"Generated background: {generated_path}")
    print(f"Mask: {saved_mask_path}")
    print(f"Composite: {output_path}")


if __name__ == "__main__":
    main()
