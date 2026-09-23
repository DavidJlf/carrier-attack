# LET THE CARRIER CARRY THE ATTACK: PRESERVING THE SUBJECT IN ADVERSARIAL IMAGE GENERATION

Anonymous ICLR submission — reproducibility code for the white-box, subject-preserving adversarial image generation pipeline. This repository is intended for reviewers and researchers to reproduce a single subject/target run, not a fixed list of our experimental cases.

## What is included

- `data/dreambooth_references/`: reference photographs for 20 DreamBooth subjects (106 images in total), with the original dataset license and provenance notes. These are LoRA training inputs, **not** the selected clean subject samples used in our experiments.
- `code/lora/`: the existing FLUX LoRA training launcher and 200-candidate subject-sample generator.
- `code/formal_pipeline/flux_auomation.py`: the original one-image automation, adapted for generic configuration. It constructs carriers and runs CRA, JIA and CIRA against the white-box classifier. Grad-CAM overlays and a contact sheet are produced for a completed run.
- `scripts/`: shell entry points and a repository-content audit.

Trained LoRA weights, selected clean samples, model checkpoints, prior results and case-specific CSV files are not distributed. **A new run does generate its own CSV/JSON metrics and images** under the configured output directory; those outputs are Git-ignored. The absence of precomputed results does not disable result generation.

## Environment

The pipeline requires a Linux NVIDIA GPU with enough memory for FLUX and the selected evaluation models. Create the base environment with `conda env create -f environment.yml && conda activate carrier-repro`. Obtain access to `black-forest-labs/FLUX.1-Kontext-dev` and install the official Diffusers repository; point `FLUX_LORA_TRAINER` to its `examples/dreambooth/train_dreambooth_lora_flux.py`. Install/configure SAM3 and provide its checkpoint via `SAM3_REPO` and `SAM3_CHECKPOINT`. Qwen2.5-VL is used by the carrier quality gate; set `QWEN_MODEL` to a local directory or accessible model ID. DINOv3 is optional for subject-preservation evaluation. Model licenses and access requirements apply separately.

## Reproduce one white-box run

Copy `configs/subject.env.example` to `configs/subject.env`, then fill in the model paths, one reference directory, subject and ImageNet source/target classes, visible carrier, and private output paths. `configs/subject.env` is ignored by Git. Commands are run from the repository root.

1. Train a subject LoRA from its included reference photographs: `bash scripts/01_train_lora.sh configs/subject.env`. Initially `EXECUTE=0` prints/checks the command; set `EXECUTE=1` to launch training.
2. Generate 200 LoRA subject-sample candidates: `bash scripts/02_generate_subject_samples.sh configs/subject.env`. This generates the images and their `metadata.csv` locally. Inspect them and choose one image containing a clear, single subject with minimal duplication/ghosting; set `SOURCE_IMAGE` to that image. The selected sample is not supplied by this repository.
3. Run carrier construction and white-box attacks: `bash scripts/03_run_carrier_attacks.sh configs/subject.env`. The script first supports a dry run (`EXECUTE=0`); with `EXECUTE=1`, it prepares the carrier, performs the quality gate, runs CRA/JIA/CIRA, and writes the per-method outputs, Grad-CAM overlays, `summary.json`, and a contact image in `OUTPUT_ROOT/RUN_NAME/`. If `DINO_MODEL` is set, it also evaluates subject preservation with DINOv3 and SAM3 and writes `subject_preservation/preservation_by_method.csv` and `preservation.json`.

`bash scripts/run_all.sh configs/subject.env` chains these stages. It stops after generating candidates until a valid `SOURCE_IMAGE` is selected; image selection is intentionally a manual quality-control step. For a subsequent attack-only run, use step 3 directly rather than retraining the LoRA.

The main path is white-box. Black-box transfer evaluation is optional: pass `--evaluate-transfer` when invoking `code/formal_pipeline/flux_auomation.py` directly. It is not run by `scripts/03_run_carrier_attacks.sh` and is not required for the paper's principal reproduction path.

The reference photographs are redistributed under the included DreamBooth dataset terms. Before publishing any derivative repository, inspect the files and license yourself with `bash scripts/audit_repository.sh`.
