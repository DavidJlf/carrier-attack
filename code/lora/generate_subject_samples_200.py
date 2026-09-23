from __future__ import annotations

import argparse
import csv
import gc
import itertools
from pathlib import Path

import torch
from diffusers import FluxPipeline


SCENES = [
    "beside a bright window", "in a green garden", "on a wooden chair",
    "on a sandy beach", "in a snowy field", "on a quiet city street",
    "on a soft blanket", "in an autumn forest", "in a clean photo studio",
    "beside colorful flowers", "inside a cardboard box", "beside a calm lake",
    "on stone steps", "in a sunny living room", "under a wooden table",
    "in a grassy field", "on a wide windowsill", "near a small fountain",
    "on a country road", "inside a cozy library",
]
SHOTS = [
    "close-up portrait", "full-body photograph", "side-profile photograph",
    "low-angle photograph", "eye-level photograph",
]
LIGHTS = [
    "soft morning light", "warm sunset light", "natural daylight",
    "soft overcast light", "cinematic evening light",
]
ACTIVITIES = [
    "sitting calmly", "walking naturally", "looking toward the camera",
    "resting comfortably", "exploring the surroundings",
]


def prompts_for(token: str, subject: str) -> list[str]:
    prompts = []
    combinations = list(itertools.product(SCENES, SHOTS, LIGHTS, ACTIVITIES))
    # A coprime stride spreads the first 200 choices across all four axes.
    chosen = (combinations[(i * 613) % len(combinations)] for i in range(200))
    for scene, shot, light, activity in chosen:
        prompts.append(
            f"a realistic {shot} of {token} {subject} {activity} {scene}, "
            f"{light}, sharp focus, natural proportions, professional photography"
        )
    if len(set(prompts)) != 200:
        raise RuntimeError("Prompt construction did not produce 200 unique prompts")
    return prompts


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--lora", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--token", required=True)
    p.add_argument("--subject", required=True)
    p.add_argument("--seed-base", type=int, required=True)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=3.5)
    args = p.parse_args()

    out = Path(args.output)
    images_dir = out / "images"
    prompts_dir = out / "prompts"
    images_dir.mkdir(parents=True, exist_ok=True)
    prompts_dir.mkdir(parents=True, exist_ok=True)
    prompts = prompts_for(args.token, args.subject)
    safe_subject = args.subject.strip().lower().replace(" ", "_")

    pipe = FluxPipeline.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, local_files_only=True
    )
    pipe.load_lora_weights(args.lora)
    pipe.enable_model_cpu_offload()

    rows = []
    for i, prompt in enumerate(prompts):
        seed = args.seed_base + i
        stem = f"{safe_subject}_{i:03d}_seed{seed}"
        image_path = images_dir / f"{stem}.png"
        prompt_path = prompts_dir / f"{stem}.txt"
        if not image_path.exists():
            image = pipe(
                prompt=prompt,
                guidance_scale=args.guidance,
                num_inference_steps=args.steps,
                height=1024,
                width=1024,
                generator=torch.Generator("cpu").manual_seed(seed),
            ).images[0]
            image.save(image_path)
        prompt_path.write_text(prompt + "\n", encoding="utf-8")
        rows.append({
            "index": i, "image": str(image_path), "prompt_file": str(prompt_path),
            "prompt": prompt, "seed": seed, "guidance_scale": args.guidance,
            "num_inference_steps": args.steps, "lora": args.lora,
        })
        print(f"[{i + 1:03d}/200] {image_path}", flush=True)

    with (out / "metadata.csv").open("w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=rows[0].keys())
        w.writeheader()
        w.writerows(rows)
    del pipe
    gc.collect()
    torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
