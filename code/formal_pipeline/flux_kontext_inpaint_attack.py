#!/usr/bin/env python3
"""Flux Kontext inpainting with adversarial (classifier-guided) noise injection.

This combines the two existing scripts:

  * flux_kontext_inpaint_step_by_step.py -- the explicit, pipeline-free
    denoising loop for FluxKontextInpaintPipeline (resolution snapping, T5/CLIP
    encoding, flow-match schedule, 2x2 latent packing, per-step mask blending).
  * sdxl_attack.py -- the "semantics-aware adversarial noise" of Section 4 of
    NotesonAttackGradient.pdf, where the victim classifier's gradient evaluated
    on the predicted clean sample x_hat_{0|t} is folded back into the model's
    per-step prediction.

Adapting Eq. (5) to Flux
------------------------
SDXL/DDIM is an epsilon-prediction model, so the notes write

    eps_tilde(x_t) = eps_theta(x_t) + c * sqrt(1 - alpha_bar_t) * grad_xt f(x_hat_{0|t}, y_tar)
    x_hat_{0|t}    = (x_t - sqrt(1 - alpha_bar_t) * eps_theta) / sqrt(alpha_bar_t)

Flux is a rectified-flow / flow-matching model: the transformer predicts a
velocity v_theta ~= noise - x_0 on the path x_t = (1 - sigma_t) * x_0 + sigma_t * noise.
The Tweedie/clean-sample estimate is therefore simply

    x_hat_{0|t} = x_t - sigma_t * v_theta(x_t)                                  (3')

and the reparameterized prediction handed to the scheduler becomes

    v_tilde(x_t) = v_theta(x_t) + c * w(sigma_t) * grad_xt f(x_hat_{0|t}, y_tar) (5')

with w(sigma_t) selectable via --attack-weight (constant / sigma_t, mirroring
the sqrt(1 - alpha_bar_t) weighting of the DDIM formulation).

Because Flux latents are 2x2-packed into tokens, the gradient is taken w.r.t.
the packed latents, which is the exact same tensor the scheduler steps -- the
packing is a pure reshape/permute, so this is equivalent to differentiating in
the unpacked latent space.

The inpaint mask is reused for the attack: by default the adversarial term is
restricted to the regenerated (white) region, since the per-step blend
overwrites everything outside the mask with the re-noised original latents
anyway.

Example
-------
    python flux_kontext_inpaint_attack.py \
        --image outputs/sam3_inpaint/base.png \
        --mask outputs/sam3_flux_inpaint/inpaint_mask.png \
        --inpaint-prompt "a dog and a fox in the background" \
        --target-class 340 --attack-scale 1000 --no-cpu-offload
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from diffusers import FluxKontextInpaintPipeline
from PIL import Image, ImageDraw, ImageFont
from torchvision.models import ResNet50_Weights, resnet50
from tqdm.auto import tqdm

DEFAULT_FLUX = "black-forest-labs/FLUX.1-Kontext-dev"

# Kontext is trained on these resolutions; the pipeline snaps the input image
# to the closest aspect ratio in this list.
PREFERRED_KONTEXT_RESOLUTIONS = [
    (672, 1568),
    (688, 1504),
    (720, 1456),
    (752, 1392),
    (800, 1328),
    (832, 1248),
    (880, 1184),
    (944, 1104),
    (1024, 1024),
    (1104, 944),
    (1184, 880),
    (1248, 832),
    (1328, 800),
    (1392, 752),
    (1456, 720),
    (1504, 688),
    (1568, 672),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    # --- inpainting (same knobs as flux_kontext_inpaint_step_by_step.py) ---
    parser.add_argument(
        "--image",
        default="outputs/sam3_inpaint/base.png",
        help="Base image to inpaint.",
    )
    parser.add_argument(
        "--mask",
        default="outputs/sam3_flux_inpaint/inpaint_mask.png",
        help="Inpaint mask; white = regenerate, black = keep.",
    )
    parser.add_argument("--inpaint-prompt", default="a dog and a fox in the background")
    parser.add_argument("--flux-model", default=DEFAULT_FLUX)
    parser.add_argument("--flux-steps", type=int, default=28)
    parser.add_argument("--flux-guidance", type=float, default=2.5)
    parser.add_argument("--strength", type=float, default=1.0)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--max-area", type=int, default=1024**2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--no-cpu-offload",
        action="store_true",
        help=(
            "Keep the whole Flux pipeline on the GPU. Recommended for the "
            "attack: the VAE backward pass runs right after its forward, which "
            "is safe under accelerate's model offload but wasteful."
        ),
    )
    parser.add_argument("--output-dir", default="outputs/flux_kontext_attack")
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )

    # --- attack (mirrors sdxl_attack.py) ---
    parser.add_argument(
        "--target-class",
        type=int,
        default=None,
        help=(
            "ImageNet class index y_tar. When set, the classifier gradient on "
            "pred_x0 is injected into the velocity prediction (Eq. 5'). Omit "
            "to run plain step-by-step inpainting."
        ),
    )
    parser.add_argument(
        "--attack-scale",
        type=float,
        default=1.0,
        help="Coefficient c scaling the classifier-induced term in Eq. (5').",
    )
    parser.add_argument(
        "--attack-weight",
        choices=("constant", "sigma"),
        default="constant",
        help=(
            "w(sigma_t) in Eq. (5'). 'constant' uses c alone (as sdxl_attack.py "
            "does in practice); 'sigma' additionally scales by sigma_t, the "
            "flow-matching analogue of sqrt(1 - alpha_bar_t)."
        ),
    )
    parser.add_argument(
        "--attack-grad-norm",
        choices=("none", "rms", "sign"),
        default="none",
        help=(
            "Optional gradient conditioning before scaling. 'rms' rescales the "
            "gradient to unit RMS (makes --attack-scale resolution/model "
            "independent); 'sign' uses its sign (FGSM-style)."
        ),
    )
    parser.add_argument(
        "--attack-start-step",
        type=int,
        default=0,
        help="First denoising step (inclusive) at which to apply the attack.",
    )
    parser.add_argument(
        "--attack-end-step",
        type=int,
        default=None,
        help="Last denoising step (inclusive). Default: all steps.",
    )
    parser.add_argument(
        "--attack-everywhere",
        action="store_true",
        help=(
            "Apply the adversarial term over the whole latent instead of only "
            "the white (regenerated) mask region."
        ),
    )
    parser.add_argument(
        "--attack-vae-checkpointing",
        action="store_true",
        default=True,
        help="Gradient-checkpoint the VAE decode used by the attack (saves VRAM).",
    )
    parser.add_argument(
        "--no-attack-vae-checkpointing",
        dest="attack_vae_checkpointing",
        action="store_false",
    )
    parser.add_argument(
        "--save-clean-baseline",
        action="store_true",
        help="Also run the identical schedule with the attack disabled.",
    )

    # --- victim classifier reporting ---
    parser.add_argument(
        "--victim-topk",
        type=int,
        default=5,
        help="ResNet-50 predictions to print for the final image.",
    )
    parser.add_argument(
        "--step-topk",
        type=int,
        default=3,
        help="ResNet-50 predictions to print for each step's pred_x0 (0 = skip).",
    )
    parser.add_argument(
        "--pred-x0-grid",
        default="pred_x0_grid.png",
        help=(
            "Filename (inside --output-dir) for a grid of the decoded pred_x0 "
            "at every step. Empty string disables it."
        ),
    )
    parser.add_argument(
        "--pred-x0-tile",
        type=int,
        default=256,
        help="Width (px) of each pred_x0 tile in the grid.",
    )
    return parser.parse_args()


def load_flux_pipeline(args: argparse.Namespace) -> FluxKontextInpaintPipeline:
    flux_dtype = torch.bfloat16 if args.device.startswith("cuda") else torch.float32
    pipe = FluxKontextInpaintPipeline.from_pretrained(
        args.flux_model, torch_dtype=flux_dtype
    )
    if args.device.startswith("cuda") and not args.no_cpu_offload:
        # The accelerate hooks fire on the modules themselves (AutoencoderKL's
        # encode/decode carry @apply_forward_hook), so the explicit loop below
        # works with offload just like the pipeline does. The attack's backward
        # pass happens immediately after the VAE forward, before any other
        # component runs, so the VAE weights are still resident on the GPU.
        pipe.enable_model_cpu_offload()
    else:
        pipe.to(args.device)

    # Only latent gradients are ever needed, never weight gradients.
    for module in (pipe.transformer, pipe.vae, pipe.text_encoder, pipe.text_encoder_2):
        module.requires_grad_(False)
    return pipe


# ---------------------------------------------------------------------------
# Helpers replicated from FluxKontextInpaintPipeline (diffusers 0.39.0)
# ---------------------------------------------------------------------------


def calculate_shift(
    image_seq_len: int,
    base_seq_len: int = 256,
    max_seq_len: int = 4096,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
) -> float:
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    b = base_shift - m * base_seq_len
    return image_seq_len * m + b


def pack_latents(
    latents: torch.Tensor, batch_size: int, num_channels: int, height: int, width: int
) -> torch.Tensor:
    """(B, C, H, W) -> (B, H/2 * W/2, C * 4): 2x2 patches become tokens."""
    latents = latents.view(batch_size, num_channels, height // 2, 2, width // 2, 2)
    latents = latents.permute(0, 2, 4, 1, 3, 5)
    return latents.reshape(batch_size, (height // 2) * (width // 2), num_channels * 4)


def unpack_latents(
    latents: torch.Tensor, height: int, width: int, vae_scale_factor: int
) -> torch.Tensor:
    """Inverse of pack_latents, with height/width given in pixels."""
    batch_size, _, channels = latents.shape
    height = 2 * (int(height) // (vae_scale_factor * 2))
    width = 2 * (int(width) // (vae_scale_factor * 2))
    latents = latents.view(batch_size, height // 2, width // 2, channels // 4, 2, 2)
    latents = latents.permute(0, 3, 1, 4, 2, 5)
    return latents.reshape(batch_size, channels // 4, height, width)


def prepare_latent_image_ids(
    height: int, width: int, device: torch.device, dtype: torch.dtype
) -> torch.Tensor:
    """Per-token (id, row, col) positions for RoPE; height/width in patches."""
    latent_image_ids = torch.zeros(height, width, 3)
    latent_image_ids[..., 1] += torch.arange(height)[:, None]
    latent_image_ids[..., 2] += torch.arange(width)[None, :]
    return latent_image_ids.reshape(height * width, 3).to(device=device, dtype=dtype)


@torch.no_grad()
def encode_prompt(
    pipe: FluxKontextInpaintPipeline, prompt: str, device: torch.device
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """CLIP pooled embedding + T5 token embeddings + all-zero text RoPE ids."""
    clip_inputs = pipe.tokenizer(
        prompt,
        padding="max_length",
        max_length=pipe.tokenizer_max_length,
        truncation=True,
        return_overflowing_tokens=False,
        return_length=False,
        return_tensors="pt",
    )
    pooled_prompt_embeds = pipe.text_encoder(
        clip_inputs.input_ids.to(device), output_hidden_states=False
    ).pooler_output
    pooled_prompt_embeds = pooled_prompt_embeds.to(
        dtype=pipe.text_encoder.dtype, device=device
    )

    t5_inputs = pipe.tokenizer_2(
        prompt,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    prompt_embeds = pipe.text_encoder_2(
        t5_inputs.input_ids.to(device), output_hidden_states=False
    )[0]
    prompt_embeds = prompt_embeds.to(dtype=pipe.text_encoder_2.dtype, device=device)

    text_ids = torch.zeros(prompt_embeds.shape[1], 3).to(
        device=device, dtype=pipe.text_encoder.dtype
    )
    return prompt_embeds, pooled_prompt_embeds, text_ids


# ---------------------------------------------------------------------------
# Victim classifier (identical treatment to sdxl_attack.py)
# ---------------------------------------------------------------------------

_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def load_victim_classifier(device: str) -> tuple[torch.nn.Module, ResNet50_Weights]:
    weights = ResNet50_Weights.DEFAULT
    victim = resnet50(weights=weights).to(device)
    victim.eval()
    victim.requires_grad_(False)
    return victim, weights


def victim_logits_from_tensor(
    image: torch.Tensor, victim: torch.nn.Module
) -> torch.Tensor:
    """Differentiable ResNet-50 forward pass on a [0, 1] image tensor.

    Mirrors ResNet50_Weights.DEFAULT.transforms() (resize to 224 + ImageNet
    normalization) using differentiable ops so gradients propagate to ``image``.
    """
    image = image.to(dtype=torch.float32)
    # Important: the attack is not working if we set antialias=False
    image = F.interpolate(
        image, size=(224, 224), mode="bilinear", align_corners=False, antialias=True
    )
    mean = torch.tensor(_IMAGENET_MEAN, device=image.device, dtype=image.dtype).view(
        1, 3, 1, 1
    )
    std = torch.tensor(_IMAGENET_STD, device=image.device, dtype=image.dtype).view(
        1, 3, 1, 1
    )
    image = (image - mean) / std
    return victim(image)


@torch.no_grad()
def classify_image(
    image: Image.Image,
    victim: torch.nn.Module,
    weights: ResNet50_Weights,
    device: str,
    topk: int,
) -> list[tuple[int, str, float]]:
    preprocess = weights.transforms()
    batch = preprocess(image).unsqueeze(0).to(device)
    probabilities = victim(batch).softmax(dim=1)[0]
    top_probabilities, top_indices = probabilities.topk(topk)

    categories = weights.meta["categories"]
    return [
        (index.item(), categories[index.item()], probability.item())
        for probability, index in zip(top_probabilities, top_indices)
    ]


def topk_from_logits(
    logits: torch.Tensor, weights: ResNet50_Weights, topk: int
) -> list[tuple[int, str, float]]:
    """Same output shape as classify_image, reusing an already-computed forward."""
    probabilities = logits.detach().float().softmax(dim=1)[0]
    top_probabilities, top_indices = probabilities.topk(topk)
    categories = weights.meta["categories"]
    return [
        (index.item(), categories[index.item()], probability.item())
        for probability, index in zip(top_probabilities, top_indices)
    ]


def make_image_grid(
    images: list[Image.Image], labels: list[str] | None = None
) -> Image.Image:
    if not images:
        raise ValueError("No images to assemble into a grid.")

    cols = math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    tile_w, tile_h = images[0].size

    label_h = 0
    font = None
    if labels is not None:
        font = ImageFont.load_default()
        label_h = 16

    cell_w = tile_w
    cell_h = tile_h + label_h
    grid = Image.new("RGB", (cols * cell_w, rows * cell_h), color=(0, 0, 0))
    draw = ImageDraw.Draw(grid)

    for index, image in enumerate(images):
        row, col = divmod(index, cols)
        x, y = col * cell_w, row * cell_h
        grid.paste(image, (x, y + label_h))
        if labels is not None and font is not None:
            draw.text((x + 2, y + 2), labels[index], fill=(255, 255, 255), font=font)
    return grid


# ---------------------------------------------------------------------------
# Adversarial noise for the flow-matching (Flux) parameterization
# ---------------------------------------------------------------------------


def decode_packed_latents(
    pipe: FluxKontextInpaintPipeline,
    packed_latents: torch.Tensor,
    height: int,
    width: int,
) -> torch.Tensor:
    """Differentiable decode of packed Flux latents to a [0, 1] image tensor."""
    latents = unpack_latents(packed_latents, height, width, pipe.vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return (image / 2 + 0.5).clamp(0, 1)


def to_pil(image: torch.Tensor) -> Image.Image:
    array = image.detach().float().cpu().permute(0, 2, 3, 1).numpy()
    return Image.fromarray((array[0] * 255).round().astype("uint8"))


def adversarial_velocity(
    pipe: FluxKontextInpaintPipeline,
    latents: torch.Tensor,
    velocity_pred: torch.Tensor,
    sigma: float,
    height: int,
    width: int,
    victim: torch.nn.Module,
    target_class: int,
    scale: float,
    weight_mode: str,
    grad_norm: str,
    mask: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Fold the classifier gradient on pred_x0 into the velocity prediction.

    Flow-matching form of Eq. (5) (see the module docstring):

        x_hat_{0|t} = x_t - sigma_t * v_theta(x_t)
        v_tilde     = v_theta + c * w(sigma_t) * grad_xt f(x_hat_{0|t}, y_tar)

    Returns the reparameterized velocity, the decoded pred_x0 image (detached,
    reused for the per-step report so it is not decoded twice) and the victim
    logits for that image.
    """
    velocity_pred = velocity_pred.detach()
    with torch.enable_grad():
        xt = latents.detach().requires_grad_(True)
        # pred_x0 / x_hat_{0|t}: the Tweedie estimate of the clean latents.
        pred_x0 = xt - sigma * velocity_pred
        image = decode_packed_latents(pipe, pred_x0, height, width)
        logits = victim_logits_from_tensor(image, victim)
        target = torch.tensor([target_class], device=logits.device)
        # f(x_hat_{0|t}, y_tar): targeted cross-entropy. Descending it along the
        # fused trajectory drives the sample toward the target class.
        loss = F.cross_entropy(logits, target)
        grad_xt = torch.autograd.grad(loss, xt)[0]

    if not torch.isfinite(grad_xt).all() or grad_xt.abs().max() == 0:
        # Usually a sign that the decode graph was severed (e.g. reentrant VAE
        # gradient checkpointing) or that bf16 underflowed the gradient.
        print(
            "  warning: classifier gradient is zero/non-finite at this step; "
            "try --no-attack-vae-checkpointing or --attack-grad-norm rms"
        )

    grad_xt = grad_xt.to(velocity_pred.dtype)
    if grad_norm == "rms":
        rms = grad_xt.float().pow(2).mean().sqrt().clamp_min(1e-12)
        grad_xt = (grad_xt.float() / rms).to(velocity_pred.dtype)
    elif grad_norm == "sign":
        grad_xt = grad_xt.sign()

    weight = scale * (float(sigma) if weight_mode == "sigma" else 1.0)
    adv_term = weight * grad_xt
    if mask is not None:
        # Outside the white region the per-step blend restores the original
        # latents anyway, so perturbing there only wastes signal.
        adv_term = adv_term * mask
    return velocity_pred + adv_term, image.detach(), logits.detach()


# ---------------------------------------------------------------------------
# Step-by-step inpainting loop with the attack fused in
# ---------------------------------------------------------------------------


@torch.no_grad()
def flux_inpaint_attack(
    pipe: FluxKontextInpaintPipeline,
    image: Image.Image,
    mask_image: Image.Image,
    args: argparse.Namespace,
    victim: torch.nn.Module,
    weights: ResNet50_Weights,
    attack_enabled: bool,
    tag: str,
) -> tuple[Image.Image, list[Image.Image], list[str]]:
    device = pipe._execution_device
    vae_scale_factor = pipe.vae_scale_factor  # 8
    multiple_of = vae_scale_factor * 2  # packing needs even latent dims
    generator = torch.Generator(device="cpu").manual_seed(args.seed)

    # --- 1. Resolution: cap by max_area, then snap the input image to the ---
    # --- closest preferred Kontext resolution (its size wins over height/width)
    height, width = args.height, args.width
    aspect_ratio = width / height
    width = round((args.max_area * aspect_ratio) ** 0.5) // multiple_of * multiple_of
    height = round((args.max_area / aspect_ratio) ** 0.5) // multiple_of * multiple_of

    image_height, image_width = pipe.image_processor.get_default_height_width(image)
    aspect_ratio = image_width / image_height
    _, image_width, image_height = min(
        (abs(aspect_ratio - w / h), w, h) for w, h in PREFERRED_KONTEXT_RESOLUTIONS
    )
    image_width = image_width // multiple_of * multiple_of
    image_height = image_height // multiple_of * multiple_of
    resized = pipe.image_processor.resize(image, image_height, image_width)
    width, height = image_width, image_height

    # PIL -> normalized (1, 3, H, W) tensor in [-1, 1]
    image_pt = pipe.image_processor.preprocess(resized, image_height, image_width)
    init_image = image_pt.to(dtype=torch.float32)

    # --- 2. Prompt encoding ---
    prompt_embeds, pooled_prompt_embeds, text_ids = encode_prompt(
        pipe, args.inpaint_prompt, device
    )
    dtype = prompt_embeds.dtype

    # --- 3. Timesteps: flow-match sigmas with resolution-dependent shift, ---
    # --- then truncate the schedule head according to strength ---
    sigmas = np.linspace(1.0, 1 / args.flux_steps, args.flux_steps)
    image_seq_len = (height // vae_scale_factor // 2) * (width // vae_scale_factor // 2)
    mu = calculate_shift(
        image_seq_len,
        pipe.scheduler.config.get("base_image_seq_len", 256),
        pipe.scheduler.config.get("max_image_seq_len", 4096),
        pipe.scheduler.config.get("base_shift", 0.5),
        pipe.scheduler.config.get("max_shift", 1.15),
    )
    pipe.scheduler.set_timesteps(sigmas=sigmas, mu=mu, device=device)
    num_inference_steps = len(pipe.scheduler.timesteps)

    init_timestep = min(num_inference_steps * args.strength, num_inference_steps)
    t_start = int(max(num_inference_steps - init_timestep, 0))
    timesteps = pipe.scheduler.timesteps[t_start * pipe.scheduler.order :]
    pipe.scheduler.set_begin_index(t_start * pipe.scheduler.order)
    if len(timesteps) < 1:
        raise ValueError(f"strength={args.strength} leaves no denoising steps")
    latent_timestep = timesteps[:1]
    # sigma_t per loop iteration; the scheduler keeps len(timesteps) + 1 sigmas.
    schedule_sigmas = pipe.scheduler.sigmas[t_start * pipe.scheduler.order :]

    # --- 4. Latents: VAE-encode the init image (mode of the latent dist), ---
    # --- noise it to the first timestep, pack everything into tokens ---
    num_channels_latents = pipe.transformer.config.in_channels // 4  # 16
    latent_height = 2 * (height // (vae_scale_factor * 2))
    latent_width = 2 * (width // (vae_scale_factor * 2))

    image_latents = pipe.vae.encode(
        init_image.to(device=device, dtype=dtype)
    ).latent_dist.mode()
    image_latents = (
        image_latents - pipe.vae.config.shift_factor
    ) * pipe.vae.config.scaling_factor

    shape = (1, num_channels_latents, latent_height, latent_width)
    # randn on the CPU generator, then moved to device (randn_tensor semantics)
    noise = torch.randn(shape, generator=generator, dtype=dtype, device="cpu").to(device)
    # sigma(t0) * noise + (1 - sigma(t0)) * image; with strength=1.0 this is pure noise
    latents = pipe.scheduler.scale_noise(image_latents, latent_timestep, noise)

    latent_ids = prepare_latent_image_ids(
        latent_height // 2, latent_width // 2, device, dtype
    )
    image_ids = prepare_latent_image_ids(
        latent_height // 2, latent_width // 2, device, dtype
    )
    image_ids[..., 0] = 1  # distinguishes the conditioning tokens from the target
    latent_ids = torch.cat([latent_ids, image_ids], dim=0)

    image_latents = pack_latents(
        image_latents, 1, num_channels_latents, latent_height, latent_width
    )
    noise = pack_latents(noise, 1, num_channels_latents, latent_height, latent_width)
    latents = pack_latents(latents, 1, num_channels_latents, latent_height, latent_width)

    # --- 5. Mask at latent resolution, expanded to the packed layout ---
    mask_condition = pipe.mask_processor.preprocess(
        mask_image, height=height, width=width, resize_mode="default", crops_coords=None
    )
    mask = torch.nn.functional.interpolate(
        mask_condition, size=(latent_height, latent_width)
    ).to(device=device, dtype=dtype)
    mask = pack_latents(
        mask.repeat(1, num_channels_latents, 1, 1),
        1,
        num_channels_latents,
        latent_height,
        latent_width,
    )

    # --- 6. Attack setup ---
    attack_end_step = (
        args.attack_end_step if args.attack_end_step is not None else len(timesteps) - 1
    )
    attack_mask = None if args.attack_everywhere else mask
    if attack_enabled:
        if args.attack_vae_checkpointing and hasattr(
            pipe.vae, "enable_gradient_checkpointing"
        ):
            pipe.vae.enable_gradient_checkpointing()
        target_label = weights.meta["categories"][args.target_class]
        print(
            f"[{tag}] attack on: target {args.target_class} ({target_label}), "
            f"c={args.attack_scale}, w={args.attack_weight}, "
            f"grad-norm={args.attack_grad_norm}, "
            f"steps {args.attack_start_step}..{attack_end_step}, "
            f"region={'whole latent' if args.attack_everywhere else 'mask only'}"
        )

    # --- 7. Denoising loop ---
    if pipe.transformer.config.guidance_embeds:
        # Kontext-dev is guidance-distilled: the scale is an input embedding
        guidance = torch.full(
            [1], args.flux_guidance, device=device, dtype=torch.float32
        ).expand(latents.shape[0])
    else:
        guidance = None

    tile_h = max(1, round(args.pred_x0_tile * height / width))
    tiles: list[Image.Image] = []
    labels: list[str] = []

    for i, t in enumerate(tqdm(timesteps, desc=f"Flux inpaint [{tag}]")):
        # target tokens and conditioning image tokens share one sequence
        latent_model_input = torch.cat([latents, image_latents], dim=1)
        timestep = t.expand(latents.shape[0]).to(latents.dtype)

        velocity_pred = pipe.transformer(
            hidden_states=latent_model_input,
            timestep=timestep / 1000,
            guidance=guidance,
            pooled_projections=pooled_prompt_embeds,
            encoder_hidden_states=prompt_embeds,
            txt_ids=text_ids,
            img_ids=latent_ids,
            joint_attention_kwargs={},
            return_dict=False,
        )[0]
        velocity_pred = velocity_pred[:, : latents.size(1)]

        sigma = float(schedule_sigmas[i])
        attack_active = (
            attack_enabled and args.attack_start_step <= i <= attack_end_step
        )

        pred_x0_image: Image.Image | None = None
        step_predictions: list[tuple[int, str, float]] = []
        if attack_active:
            velocity_pred, decoded, logits = adversarial_velocity(
                pipe=pipe,
                latents=latents,
                velocity_pred=velocity_pred,
                sigma=sigma,
                height=height,
                width=width,
                victim=victim,
                target_class=args.target_class,
                scale=args.attack_scale,
                weight_mode=args.attack_weight,
                grad_norm=args.attack_grad_norm,
                mask=attack_mask,
            )
            if args.step_topk > 0:
                # Reuse the decode/forward the attack already performed.
                pred_x0_image = to_pil(decoded)
                step_predictions = topk_from_logits(logits, weights, args.step_topk)
        elif args.step_topk > 0 or args.pred_x0_grid:
            pred_x0 = latents - sigma * velocity_pred
            pred_x0_image = to_pil(decode_packed_latents(pipe, pred_x0, height, width))
            if args.step_topk > 0:
                step_predictions = classify_image(
                    pred_x0_image, victim, weights, args.device, args.step_topk
                )

        if step_predictions:
            summary = ", ".join(
                f"{label} ({probability:.3f})"
                for _, label, probability in step_predictions
            )
            print(f"[{tag}] step {i:02d} (t={int(t)}, sigma={sigma:.4f}) pred_x0 -> {summary}")

        if args.pred_x0_grid and pred_x0_image is not None:
            tiles.append(
                pred_x0_image.resize((args.pred_x0_tile, tile_h), Image.LANCZOS)
            )
            top_label = step_predictions[0][1] if step_predictions else ""
            labels.append(f"step {i} (t={int(t)}) {top_label}")

        latents = pipe.scheduler.step(velocity_pred, t, latents, return_dict=False)[0]

        # keep the unmasked region: re-noise the init latents to the next
        # timestep and blend them back in (black mask pixels = keep)
        init_latents_proper = image_latents
        if i < len(timesteps) - 1:
            noise_timestep = timesteps[i + 1]
            init_latents_proper = pipe.scheduler.scale_noise(
                init_latents_proper, torch.tensor([noise_timestep]), noise
            )
        latents = (1 - mask) * init_latents_proper + mask * latents

    # --- 8. Decode ---
    latents = unpack_latents(latents, height, width, vae_scale_factor)
    latents = (latents / pipe.vae.config.scaling_factor) + pipe.vae.config.shift_factor
    image_out = pipe.vae.decode(latents.to(pipe.vae.dtype), return_dict=False)[0]
    return pipe.image_processor.postprocess(image_out, output_type="pil")[0], tiles, labels


def report(
    name: str,
    image: Image.Image,
    victim: torch.nn.Module,
    weights: ResNet50_Weights,
    args: argparse.Namespace,
) -> None:
    predictions = classify_image(image, victim, weights, args.device, args.victim_topk)
    print(f"ResNet-50 predictions for {name}:")
    for rank, (index, label, probability) in enumerate(predictions, start=1):
        print(f"  {rank}. {label} (class {index}): {probability:.4f}")
    if args.target_class is not None:
        ranks = [i for i, (idx, _, _) in enumerate(predictions, 1) if idx == args.target_class]
        target_label = weights.meta["categories"][args.target_class]
        hit = f"rank {ranks[0]}" if ranks else f"not in top-{args.victim_topk}"
        print(f"  target {args.target_class} ({target_label}): {hit}")


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    image = Image.open(args.image).convert("RGB")
    mask_image = Image.open(args.mask)

    victim, weights = load_victim_classifier(args.device)

    print(f"Loading {args.flux_model} ...")
    pipe = load_flux_pipeline(args)

    attack_enabled = args.target_class is not None
    if not attack_enabled:
        print("No --target-class given: running plain step-by-step inpainting.")

    result, tiles, labels = flux_inpaint_attack(
        pipe,
        image,
        mask_image,
        args,
        victim,
        weights,
        attack_enabled=attack_enabled,
        tag="attack" if attack_enabled else "clean",
    )
    result_path = out_dir / ("attacked.png" if attack_enabled else "inpainted.png")
    result.save(result_path)
    print(f"Saved result to {result_path}")

    if args.pred_x0_grid and tiles:
        grid_path = out_dir / args.pred_x0_grid
        make_image_grid(tiles, labels=labels).save(grid_path)
        print(f"Saved pred_x0 grid to {grid_path}")

    report(result_path.name, result, victim, weights, args)

    if args.save_clean_baseline and attack_enabled:
        baseline, baseline_tiles, baseline_labels = flux_inpaint_attack(
            pipe,
            image,
            mask_image,
            args,
            victim,
            weights,
            attack_enabled=False,
            tag="clean",
        )
        baseline_path = out_dir / "clean.png"
        baseline.save(baseline_path)
        print(f"Saved clean baseline to {baseline_path}")
        if args.pred_x0_grid and baseline_tiles:
            grid_path = out_dir / f"clean_{args.pred_x0_grid}"
            make_image_grid(baseline_tiles, labels=baseline_labels).save(grid_path)
            print(f"Saved clean pred_x0 grid to {grid_path}")
        report(baseline_path.name, baseline, victim, weights, args)

        diff = np.abs(
            np.asarray(result).astype(np.int32) - np.asarray(baseline).astype(np.int32)
        )
        print(
            f"attacked vs clean: max|diff|={int(diff.max())}, "
            f"mean|diff|={diff.mean():.4f}, "
            f"pixels differing={float((diff > 0).any(axis=-1).mean() * 100.0):.4f}%"
        )


if __name__ == "__main__":
    main()
