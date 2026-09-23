#!/usr/bin/env python3
"""CRA: global FLUX classifier attack with rectified-flow inversion.

This is a FLUX-native port of the experiment structure in ``sdxl_attack_input.py``.
It is deliberately independent from the FLUX inpainting attack:

* prompt mode starts from a random FLUX latent;
* input mode VAE-encodes an existing composite and performs full rectified-flow
  (Euler/ODE) inversion to obtain a clean pivot trajectory;
* the denoising velocity is modified by a differentiable ImageNet classifier;
* attack guidance is applied globally; no protected-mask route is implemented.

FLUX Kontext is guidance-distilled and has no SDXL unconditional CFG branch.
Consequently, ``--use_null_text`` is accepted as a compatibility switch but maps
to per-step pivot-velocity correction, which serves the same reconstruction goal
without pretending to optimize a nonexistent unconditional text embedding.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxKontextPipeline
from PIL import Image
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm


DEFAULT_MODEL = "black-forest-labs/FLUX.1-Kontext-dev"
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass
class StepRecord:
    step: int
    sigma: float
    attacked: bool
    scale: float
    target_probability: Optional[float]
    target_rank: Optional[int]
    loss: Optional[float]
    grad_rms: Optional[float]
    update_rms: Optional[float]


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def safe_name(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", text).strip("_") or "run"


def parse_classes(value: str) -> list[int]:
    if not value.strip():
        return []
    return [int(part.strip()) for part in value.split(",") if part.strip()]


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    slope = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * slope + base_shift - slope * base_seq_len


def pack_latents(latents: torch.Tensor) -> torch.Tensor:
    b, c, h, w = latents.shape
    latents = latents.view(b, c, h // 2, 2, w // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(b, (h // 2) * (w // 2), c * 4)


def unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
) -> torch.Tensor:
    b, _, channels = latents.shape
    h = 2 * (height // (vae_scale_factor * 2))
    w = 2 * (width // (vae_scale_factor * 2))
    latents = latents.view(b, h // 2, w // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(b, channels // 4, h, w)


def latent_ids(height: int, width: int, device, dtype) -> torch.Tensor:
    ids = torch.zeros(height, width, 3, device=device, dtype=dtype)
    ids[..., 1] = torch.arange(height, device=device, dtype=dtype)[:, None]
    ids[..., 2] = torch.arange(width, device=device, dtype=dtype)[None, :]
    return ids.reshape(height * width, 3)


@torch.no_grad()
def encode_prompt(pipe: FluxKontextPipeline, prompt: str, device):
    clip = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_tensors="pt",
    )
    pooled = pipe.text_encoder(clip.input_ids.to(device)).pooler_output
    pooled = pooled.to(device=device, dtype=pipe.text_encoder.dtype)
    t5 = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt",
    )
    embeds = pipe.text_encoder_2(t5.input_ids.to(device))[0]
    embeds = embeds.to(device=device, dtype=pipe.text_encoder_2.dtype)
    text_ids = torch.zeros(embeds.shape[1], 3, device=device, dtype=embeds.dtype)
    return embeds, pooled, text_ids


def load_image(path: str, height: int, width: int) -> Image.Image:
    image = Image.open(path).convert("RGB")
    return image.resize((width, height), Image.Resampling.LANCZOS)


def encode_image(pipe, image: Image.Image, device, dtype) -> torch.Tensor:
    tensor = pipe.image_processor.preprocess(image, image.height, image.width)
    tensor = tensor.to(device=device, dtype=dtype)
    latent = pipe.vae.encode(tensor).latent_dist.mode()
    latent = (latent - pipe.vae.config.shift_factor) * pipe.vae.config.scaling_factor
    return pack_latents(latent)


def decode_latents(pipe, packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    latent = unpack_latents(packed, height, width, pipe.vae_scale_factor)
    latent = latent / pipe.vae.config.scaling_factor + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latent.to(pipe.vae.dtype), return_dict=False)[0]
    return (image.float() / 2 + 0.5).clamp(0, 1)


def tensor_to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().cpu()[0].permute(1, 2, 0).numpy()
    return Image.fromarray(np.clip(array * 255.0, 0, 255).round().astype(np.uint8))


def model_velocity(
    pipe,
    latents,
    context_latents,
    sigma: float,
    prompt_embeds,
    pooled_prompt_embeds,
    text_ids,
    image_ids,
    guidance: float,
) -> torch.Tensor:
    hidden = torch.cat([latents, context_latents], dim=1)
    timestep = torch.full(
        (latents.shape[0],), sigma, device=latents.device, dtype=latents.dtype
    )
    guidance_tensor = None
    if pipe.transformer.config.guidance_embeds:
        guidance_tensor = torch.full(
            (latents.shape[0],), guidance, device=latents.device, dtype=torch.float32
        )
    velocity = pipe.transformer(
        hidden_states=hidden,
        timestep=timestep,
        guidance=guidance_tensor,
        pooled_projections=pooled_prompt_embeds,
        encoder_hidden_states=prompt_embeds,
        txt_ids=text_ids,
        img_ids=image_ids,
        joint_attention_kwargs={},
        return_dict=False,
    )[0]
    return velocity[:, : latents.shape[1]]


def make_schedule(pipe, num_steps: int, height: int, width: int, device) -> list[float]:
    seq_len = (height // pipe.vae_scale_factor // 2) * (
        width // pipe.vae_scale_factor // 2
    )
    mu = calculate_shift(
        seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(num_steps, device=device, mu=mu)
    return [float(x) for x in pipe.scheduler.sigmas.detach().cpu().tolist()]


@torch.no_grad()
def rf_invert(
    pipe,
    z0,
    context,
    descending_sigmas,
    prompt_data,
    ids,
    guidance,
) -> list[torch.Tensor]:
    """Euler ODE inversion from sigma=0 to sigma=max; returns descending pivots."""
    ascending = list(reversed(descending_sigmas))
    current = z0.detach().clone()
    pivots_ascending = [current.detach().clone()]
    embeds, pooled, text_ids = prompt_data
    for index in tqdm(range(len(ascending) - 1), desc="FLUX RF inversion"):
        sigma = ascending[index]
        next_sigma = ascending[index + 1]
        velocity = model_velocity(
            pipe, current, context, sigma, embeds, pooled, text_ids, ids, guidance
        )
        current = current + (next_sigma - sigma) * velocity
        pivots_ascending.append(current.detach().clone())
    return list(reversed(pivots_ascending))


def classifier(device):
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights).to(device).eval()
    model.requires_grad_(False)
    return model, weights


def classifier_logits(image: torch.Tensor, model) -> torch.Tensor:
    x = F.interpolate(
        image.float(), (232, 232), mode="bilinear", align_corners=False, antialias=True
    )
    x = x[:, :, 4:228, 4:228]
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return model((x - mean) / std)


def evaluate(image: Image.Image, model, weights, device, target: int) -> dict:
    batch = weights.transforms()(image).unsqueeze(0).to(device)
    probs = model(batch).softmax(1)[0]
    order = probs.argsort(descending=True)
    categories = weights.meta["categories"]
    top5 = [
        {"class": int(i), "label": categories[int(i)], "probability": float(probs[i])}
        for i in order[:5]
    ]
    rank = int((order == target).nonzero(as_tuple=False)[0, 0]) + 1
    return {
        "target_class": target,
        "target_label": categories[target],
        "target_probability": float(probs[target]),
        "target_rank": rank,
        "top5": top5,
    }


def effective_scale(args, step: int) -> float:
    start, stop = args.attack_start_step, args.attack_stop_step
    if args.scale_schedule == "constant" or stop <= start:
        return args.classifier_scale
    progress = min(1.0, max(0.0, (step - start) / (stop - start)))
    early = args.classifier_scale if args.classifier_scale_early is None else args.classifier_scale_early
    late = args.classifier_scale if args.classifier_scale_late is None else args.classifier_scale_late
    if args.scale_schedule == "linear":
        return early + progress * (late - early)
    return late + 0.5 * (early - late) * (1 + math.cos(math.pi * progress))


def attack_velocity(
    pipe,
    latents,
    velocity,
    sigma,
    height,
    width,
    victim,
    target_class,
    source_classes,
    source_weight,
    scale,
    objective,
    attack_sign,
    normalize_grad,
    background_mask,
    max_update_norm,
):
    base_velocity = velocity.detach()
    with torch.enable_grad():
        xt = latents.detach().requires_grad_(True)
        pred_x0 = xt - sigma * base_velocity
        decoded = decode_latents(pipe, pred_x0, height, width)
        logits = classifier_logits(decoded, victim)
        log_probs = logits.log_softmax(1)
        ce = F.cross_entropy(logits, torch.tensor([target_class], device=logits.device))
        source_term = torch.tensor(0.0, device=logits.device)
        if source_classes:
            source_term = torch.logsumexp(log_probs[:, source_classes], dim=1).mean()
        if objective == "ce":
            loss = ce + source_weight * source_term
        else:
            loss = log_probs[:, target_class].mean() - source_weight * source_term
        grad = torch.autograd.grad(loss, xt)[0]
    if normalize_grad:
        grad = grad / grad.float().pow(2).mean().sqrt().clamp_min(1e-12)
    update = attack_sign * scale * sigma * grad.to(base_velocity.dtype)
    if background_mask is not None:
        update = update * background_mask
    if max_update_norm and max_update_norm > 0:
        norm = update.float().flatten(1).norm(dim=1).clamp_min(1e-12)
        factor = (max_update_norm / norm).clamp(max=1.0).to(update.dtype)
        update = update * factor[:, None, None]
    probs = logits.detach().softmax(1)[0]
    rank = int((probs.argsort(descending=True) == target_class).nonzero()[0, 0]) + 1
    stats = {
        "probability": float(probs[target_class]),
        "rank": rank,
        "loss": float(loss.detach()),
        "grad_rms": float(grad.detach().float().pow(2).mean().sqrt()),
        "update_rms": float(update.detach().float().pow(2).mean().sqrt()),
    }
    return base_velocity + update, stats


def run_forward(
    pipe,
    initial,
    context,
    sigmas,
    prompt_data,
    ids,
    args,
    victim,
    pivots=None,
    attack=True,
):
    latents = initial.detach().clone()
    embeds, pooled, text_ids = prompt_data
    records = []
    progress_name = (
        "FLUX adversarial denoising" if attack else "FLUX clean reconstruction"
    )
    for step in tqdm(range(len(sigmas) - 1), desc=progress_name):
        sigma, next_sigma = sigmas[step], sigmas[step + 1]
        velocity = model_velocity(
            pipe, latents, context, sigma, embeds, pooled, text_ids, ids, args.guidance_scale
        )
        # FLUX replacement for null-text reconstruction: correct the model
        # velocity toward the exact clean pivot transition obtained by inversion.
        if pivots is not None and args.pivot_correction > 0:
            delta = next_sigma - sigma
            pivot_velocity = (pivots[step + 1] - pivots[step]) / delta
            velocity = velocity + args.pivot_correction * (pivot_velocity - velocity)
        active = (
            attack
            and args.classifier_scale > 0
            and args.attack_start_step <= step + 1 <= args.attack_stop_step
        )
        stats = None
        scale = effective_scale(args, step + 1) if active else 0.0
        if active:
            velocity, stats = attack_velocity(
                pipe,
                latents,
                velocity,
                sigma,
                args.height,
                args.width,
                victim,
                args.target_class,
                parse_classes(args.source_classes),
                args.source_suppress_weight,
                scale,
                args.objective,
                args.attack_sign,
                args.normalize_grad,
                None,
                args.max_update_norm if args.clip_mode == "l2" else None,
            )
        latents = latents + (next_sigma - sigma) * velocity
        records.append(
            StepRecord(
                step=step + 1,
                sigma=sigma,
                attacked=active,
                scale=scale,
                target_probability=None if stats is None else stats["probability"],
                target_rank=None if stats is None else stats["rank"],
                loss=None if stats is None else stats["loss"],
                grad_rms=None if stats is None else stats["grad_rms"],
                update_rms=None if stats is None else stats["update_rms"],
            )
        )
    return latents, records


def save_records(records: list[StepRecord], path: Path) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(asdict(records[0]).keys()))
        writer.writeheader()
        writer.writerows(asdict(record) for record in records)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        description="FLUX RF-inversion input attack and direct prompt attack.",
    )
    parser.add_argument("--model_id", "--flux-model", default=DEFAULT_MODEL)
    parser.add_argument("--local_files_only", action="store_true", default=True)
    parser.add_argument("--allow_download", dest="local_files_only", action="store_false")
    parser.add_argument("--input_image")
    parser.add_argument("--single_image_attack", action="store_true")
    parser.add_argument("--prompt", default="a realistic photograph")
    parser.add_argument(
        "--inversion_prompt",
        help="Optional source prompt used only for RF inversion; forward generation uses --prompt.",
    )
    parser.add_argument("--negative_prompt", default="", help="Compatibility only; FLUX Kontext has no negative CFG branch.")
    parser.add_argument("--target_class", type=int, default=407)
    parser.add_argument("--target_label", default="ambulance")
    parser.add_argument("--source_classes", default="281,282,285")
    parser.add_argument("--num_steps", type=int, default=30)
    parser.add_argument("--inversion_steps", type=int, default=30, help="Accepted for SDXL CLI compatibility; FLUX uses the forward schedule length.")
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--guidance_scale", type=float, default=2.5)
    parser.add_argument("--inversion_guidance_scale", type=float, default=2.5)
    parser.add_argument("--recon_guidance_scale", type=float, default=1.0)
    parser.add_argument("--pivot_correction", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--use_null_text", action="store_true", help="Compatibility alias: enable FLUX pivot-velocity correction, not SDXL null-text embeddings.")
    parser.add_argument("--null_text_steps", type=int, default=30, help="Compatibility-only metadata.")
    parser.add_argument("--null_text_lr", type=float, default=0.01, help="Compatibility-only metadata.")
    parser.add_argument("--classifier_scale", "--attack-scale", type=float, default=1.0)
    parser.add_argument("--classifier_scale_early", type=float)
    parser.add_argument("--classifier_scale_late", type=float)
    parser.add_argument("--scale_schedule", choices=("constant", "linear", "cosine"), default="constant")
    parser.add_argument("--attack_start_step", type=int, default=3)
    parser.add_argument("--attack_stop_step", "--attack_end_step", type=int, default=20)
    parser.add_argument("--objective", choices=("ce", "logprob_direct"), default="ce")
    parser.add_argument("--attack_sign", type=float, default=1.0)
    parser.add_argument("--source_suppress_weight", type=float, default=0.0)
    parser.add_argument("--normalize_grad", action="store_true")
    parser.add_argument("--max_update_norm", type=float, default=10.0)
    parser.add_argument("--clip_mode", default="none", choices=("none", "l2"), help="Compatibility name; l2 uses --max_update_norm.")
    parser.add_argument("--save_path", default="./outputs_flux_attack")
    parser.add_argument("--version_tag", default="flux_attack")
    parser.add_argument("--skip_clean_baseline", action="store_true")
    parser.add_argument(
        "--generation_only",
        action="store_true",
        help="Run inversion/forward editing without classifier attack and save composite.png.",
    )
    parser.add_argument("--no_cpu_offload", action="store_true")
    parser.add_argument("--fp32", action="store_true")
    parser.add_argument("--lora_path", help="Optional FLUX-compatible LoRA only; SDXL LoRAs are incompatible.")
    parser.add_argument("--lora_scale", type=float, default=1.0)
    parser.add_argument("--use_unet_lora", action="store_true", help="Accepted for compatibility; FLUX has a Transformer, not a UNet.")
    parser.add_argument("--use_text_encoder_lora", action="store_true", help="Accepted for compatibility; use a FLUX-native adapter.")
    parser.add_argument("--unet_lora_scale", type=float, default=1.0)
    parser.add_argument("--text_encoder_lora_scale", type=float, default=1.0)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    input_mode = bool(args.single_image_attack or args.input_image)
    if args.single_image_attack and not args.input_image:
        raise ValueError("--single_image_attack requires --input_image")
    if args.lora_path and (args.use_unet_lora or args.use_text_encoder_lora):
        print("[WARN] SDXL LoRA component flags are ignored; loading the path as a FLUX adapter.")
    if args.pivot_correction is None:
        args.pivot_correction = args.recon_guidance_scale if (args.use_null_text or input_mode) else 0.0
    if args.negative_prompt:
        print("[WARN] --negative_prompt is recorded but not used by guidance-distilled FLUX Kontext.")
    if input_mode and args.inversion_steps != args.num_steps:
        print("[WARN] FLUX RF inversion must share the forward sigma grid; using --num_steps for both.")

    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.float32 if args.fp32 or device.type == "cpu" else torch.bfloat16
    output = Path(args.save_path) / safe_name(args.version_tag)
    output.mkdir(parents=True, exist_ok=True)
    images_dir = output / "images"
    images_dir.mkdir(parents=True, exist_ok=True)
    run_name = safe_name(args.version_tag)
    with (output / "config.json").open("w", encoding="utf-8") as handle:
        json.dump(vars(args), handle, indent=2, ensure_ascii=False)

    print(f"Loading {args.model_id} ({dtype})")
    pipe = FluxKontextPipeline.from_pretrained(
        args.model_id,
        torch_dtype=dtype,
        local_files_only=args.local_files_only,
    )
    if device.type == "cuda" and not args.no_cpu_offload:
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(device)
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False)
    if args.lora_path:
        lora_path = Path(args.lora_path).expanduser()
        pipe.load_lora_weights(
            str(lora_path.parent),
            weight_name=lora_path.name,
            adapter_name="attack_lora",
        )
        pipe.set_adapters(["attack_lora"], adapter_weights=[args.lora_scale])

    execution_device = pipe._execution_device
    prompt_data = encode_prompt(pipe, args.prompt, execution_device)
    inversion_prompt_data = encode_prompt(
        pipe, args.inversion_prompt or args.prompt, execution_device
    )
    sigmas = make_schedule(pipe, args.num_steps, args.height, args.width, execution_device)
    latent_h = 2 * (args.height // 16)
    latent_w = 2 * (args.width // 16)
    target_ids = latent_ids(latent_h // 2, latent_w // 2, execution_device, dtype)
    context_ids = target_ids.clone()
    context_ids[..., 0] = 1
    ids = torch.cat([target_ids, context_ids], dim=0)

    pivots = None
    if input_mode:
        source_image = load_image(args.input_image, args.height, args.width)
        source_image.save(output / "input.png")
        source_image.save(images_dir / f"{run_name}_SOURCE_resized.png")
        z0 = encode_image(pipe, source_image, execution_device, dtype)
        context = z0.detach().clone()
        pivots = rf_invert(
            pipe,
            z0,
            context,
            sigmas,
            inversion_prompt_data,
            ids,
            args.inversion_guidance_scale,
        )
        initial = pivots[0]
        torch.save(
            {"sigmas": sigmas, "pivots": [x.detach().cpu() for x in pivots]},
            output / "clean_pivot_trajectory.pt",
        )
    else:
        generator = torch.Generator(device="cpu").manual_seed(args.seed)
        shape = (1, (latent_h // 2) * (latent_w // 2), 64)
        initial = torch.randn(shape, generator=generator, dtype=dtype).to(execution_device)
        # Kontext requires conditioning-image tokens. A zero latent is used only
        # for direct prompt compatibility; input mode remains the recommended path.
        context = torch.zeros_like(initial)

    victim, weights = classifier(execution_device)
    if input_mode and not args.skip_clean_baseline:
        clean_latents, clean_records = run_forward(
            pipe, initial, context, sigmas, prompt_data, ids, args, victim,
            pivots=pivots, attack=False
        )
        clean_image = tensor_to_pil(decode_latents(pipe, clean_latents, args.height, args.width))
        clean_image.save(output / "clean_reconstruction.png")
        clean_image.save(images_dir / f"{run_name}_CLEAN_seed{args.seed}.png")
        with (output / "clean_eval.json").open("w", encoding="utf-8") as handle:
            json.dump(evaluate(clean_image, victim, weights, execution_device, args.target_class), handle, indent=2)
        save_records(clean_records, output / "clean_steps.csv")

    attacked_latents, records = run_forward(
        pipe, initial, context, sigmas, prompt_data, ids, args, victim,
        pivots=pivots, attack=not args.generation_only
    )
    attacked = tensor_to_pil(decode_latents(pipe, attacked_latents, args.height, args.width))
    image_name = "composite.png" if args.generation_only else "attacked.png"
    attacked.save(output / image_name)
    if args.generation_only:
        standard_image_name = f"{run_name}_COMPOSITE_seed{args.seed}.png"
    else:
        standard_image_name = (
            f"{run_name}_ATTACK_target{args.target_class}_{safe_name(args.target_label)}"
            f"_scale{args.classifier_scale}_start{args.attack_start_step}"
            f"_stop{args.attack_stop_step}_clip{args.clip_mode}"
            f"_obj{args.objective}_srcw{args.source_suppress_weight}_seed{args.seed}.png"
        )
    attacked.save(images_dir / standard_image_name)
    result = evaluate(attacked, victim, weights, execution_device, args.target_class)
    with (output / "attack_eval.json").open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)
    save_records(records, output / "attack_steps.csv")
    print(json.dumps(result, indent=2))
    print(f"Saved run to {output}")


if __name__ == "__main__":
    main()
