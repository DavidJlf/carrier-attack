#!/usr/bin/env python3
"""Run CRA with partial FLUX inversion and global classifier guidance."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from types import ModuleType


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_BASE = SCRIPT_DIR / "cra_flux_attack.py"


def parse_wrapper_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--base-flux-attack", type=Path, default=DEFAULT_BASE)
    parser.add_argument("--full-steps", type=int, default=50)
    parser.add_argument("--return-to-progress", type=int, default=30)
    parser.add_argument("--run-name", required=True)
    parser.add_argument("--output-root", type=Path, default=SCRIPT_DIR / "outputs")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_known_args()


def load_base(path: Path) -> ModuleType:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"FLUX attack entrypoint not found: {path}")
    spec = importlib.util.spec_from_file_location("flux_attack_base_readonly", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def main() -> None:
    wrapper, base_tokens = parse_wrapper_args()
    if wrapper.full_steps < 2:
        raise ValueError("--full-steps must be at least 2")
    if not 0 < wrapper.return_to_progress < wrapper.full_steps:
        raise ValueError("--return-to-progress must be in (0, full_steps)")

    # 50 -> 30 -> 50 counts completed generation progress.
    # Therefore the image is inverted across only the last 20 intervals, then
    # attacked while traversing those same 20 low-noise intervals forward.
    partial_intervals = wrapper.full_steps - wrapper.return_to_progress
    base = load_base(wrapper.base_flux_attack)
    original_make_schedule = base.make_schedule

    # The base attack uses model-level CPU offload, whose transformer-attention
    # peak can still fill a 48 GiB GPU at 768px. Redirect only this imported
    # process to submodule-level sequential offload.
    pipeline_class = base.FluxKontextPipeline
    original_model_offload = pipeline_class.enable_model_cpu_offload
    original_set_adapters = pipeline_class.set_adapters

    def defer_offload_until_lora_is_loaded(self, *args, **kwargs):
        del args, kwargs
        print("Low-VRAM mode: deferring offload until after LoRA setup")

    def set_adapters_then_freeze_and_offload(self, *args, **kwargs):
        result = original_set_adapters(self, *args, **kwargs)
        # The base script freezes modules before load_lora_weights(), so newly
        # injected adapter parameters may otherwise remain trainable and make
        # every RF transformer call retain an unnecessary autograd graph.
        for module in (
            self.transformer,
            self.vae,
            self.text_encoder,
            self.text_encoder_2,
        ):
            module.requires_grad_(False)
        trainable = sum(p.numel() for component in self.components.values()
                        if hasattr(component, "parameters")
                        for p in component.parameters() if p.requires_grad)
        if trainable:
            raise RuntimeError(f"Inference pipeline still has {trainable} trainable parameters")
        print("LoRA loaded and frozen; sequential CPU offload enabled")
        self.enable_sequential_cpu_offload()
        return result

    pipeline_class.enable_model_cpu_offload = defer_offload_until_lora_is_loaded
    pipeline_class.set_adapters = set_adapters_then_freeze_and_offload

    def make_partial_schedule(pipe, num_steps, height, width, device):
        full_sigmas = original_make_schedule(
            pipe, wrapper.full_steps, height, width, device
        )
        expected = wrapper.full_steps + 1
        if len(full_sigmas) != expected:
            raise RuntimeError(
                f"Expected {expected} sigma boundaries, got {len(full_sigmas)}"
            )
        partial = full_sigmas[-(partial_intervals + 1):]
        print(
            "FLUX partial RF inversion grid: "
            f"full={wrapper.full_steps} intervals, "
            f"progress {wrapper.full_steps}->{wrapper.return_to_progress}, "
            f"partial={partial_intervals} intervals, "
            f"max_partial_sigma={partial[0]:.6f}"
        )
        return partial

    base.make_schedule = make_partial_schedule
    output_root = wrapper.output_root.expanduser().resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # The base parser retains --inversion_steps for SDXL CLI compatibility, but
    # RF inversion and forward attack both consume our patched partial grid.
    forced = [
        "--num_steps", str(wrapper.full_steps),
        "--inversion_steps", str(wrapper.full_steps),
        "--attack_start_step", "1",
        "--attack_stop_step", str(partial_intervals),
        "--save_path", str(output_root),
        "--version_tag", wrapper.run_name,
    ]
    sys.argv = [sys.argv[0], *base_tokens, *forced]

    print("CRA FLUX finite-return configuration:")
    print(f"  generation progress: {wrapper.full_steps}->{wrapper.return_to_progress}->{wrapper.full_steps}")
    print(f"  RF inversion intervals: {partial_intervals}")
    print(f"  attacked forward intervals: 1..{partial_intervals}")
    print(f"  output: {output_root / wrapper.run_name}")
    if wrapper.dry_run:
        print("Base arguments:", " ".join(sys.argv[1:]))
        print("Dry run only; FLUX was not loaded.")
        return
    try:
        base.main()
    finally:
        pipeline_class.enable_model_cpu_offload = original_model_offload
        pipeline_class.set_adapters = original_set_adapters


if __name__ == "__main__":
    main()
