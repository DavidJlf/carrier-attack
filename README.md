# Carrier Reproducibility

Code-only reproduction package for subject-specific FLUX LoRA training, subject-sample generation, carrier construction, and the CRA/CIRA/JIA attacks.

This repository intentionally contains no images, DreamBooth references, trained LoRA weights, experiment tables, result CSV/JSON files, contact sheets, or model checkpoints. Runtime outputs are written under `outputs/`, which is ignored by Git.

## Pipeline

### 1. Train a subject LoRA and generate subject samples

Prepare one directory containing the provided DreamBooth reference images for each subject. Repeat this stage for all 20 subjects used by the experiment. The references and resulting 20 subject-specific LoRAs remain local and are not distributed in this repository.

```bash
cp configs/subject.env.example configs/subject.env
# Edit every path and label in configs/subject.env.
bash scripts/01_train_lora.sh configs/subject.env
bash scripts/02_generate_subject_samples.sh configs/subject.env
```

The second command generates 200 candidate images for the configured subject. Repeat it for each trained subject LoRA. For later carrier construction, select a clean subject sample with one dominant subject, little or no ghosting, no duplicated body parts, and minimal distracting objects. In the paper and repository documentation these generated clean inputs are called **subject samples**.

Both scripts are dry-run by default. Set `EXECUTE=1` in the local config only after checking the printed command, model paths, GPU memory, and output directory.

### 2. Construct a carrier and run CRA/CIRA/JIA

```bash
bash scripts/03_run_carrier_attacks.sh configs/subject.env
```

The public method names are used throughout:

- **CRA**: carrier composite followed by global finite-return attack.
- **JIA**: joint inpainting attack.
- **CIRA**: clean inpainting followed by global finite-return attack.

SAM3 is used only to obtain the clean-image subject mask required for carrier construction and inpainting. None of the three attacks uses protected-subject-gradient or protected-trajectory optimization. The orchestration creates only `cra/`, `jia/`, and `cira/` method directories; no `protected_mask/` directory is created.

To inspect commands without loading models or running GPU inference, leave `EXECUTE=0`. To run only selected attacks, set `METHODS=cra`, `METHODS=cra,cira`, or another comma-separated subset.

## Environment

Recommended: Linux, NVIDIA GPU, CUDA 12.x, Python 3.10 or 3.11, and a recent PyTorch build matching the installed CUDA driver.

```bash
conda env create -f environment.yml
conda activate carrier-repro
accelerate config
```

External gated model access may be required for FLUX and SAM3. Store Hugging Face tokens outside the repository. The LoRA launcher expects the official Diffusers `train_dreambooth_lora_flux.py`; set `FLUX_LORA_TRAINER` in the local config to its path.

## Repository safety

Before every push, run:

```bash
bash scripts/audit_repository.sh
```

The audit rejects tracked image files, archives, checkpoints, LoRA weights, CSV/Excel tables, result JSON files, caches, secrets, and generated output directories.

## Main entry points

- `training/train_flux_lora.py`: validated launcher for the official FLUX DreamBooth LoRA trainer.
- `training/generate_subject_samples.py`: deterministic 200-image subject-sample generator.
- `src/carrier/run_pipeline.py`: carrier construction and CRA/CIRA/JIA orchestration.
- `src/carrier/cra_attack.py`: global FLUX finite-return attack implementation.
- `src/carrier/inpainting_attack.py`: JIA clean/attack implementation used by JIA and CIRA preparation.
- `src/carrier/sam3_single.py`: clean-image SAM3 subject-mask extraction.
