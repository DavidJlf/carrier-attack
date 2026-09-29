# LET THE CARRIER CARRY THE ATTACK: PRESERVING THE SUBJECT IN ADVERSARIAL IMAGE GENERATION

The pipeline reproduces a white-box, subject-preserving adversarial image generation run for a chosen subject and target class.

## What is included

- `data/dreambooth_references/`: reference images for 20 DreamBooth subjects (106 images total).
- `code/lora/`: FLUX LoRA training and subject-sample generation.
- `code/formal_pipeline/flux_automation.py`: carrier construction, CRA/JIA/CIRA, Grad-CAM and contact-image generation. `carrier_catalog.py` selects the Non-Target Carrier; `hybrid_carrier_prompts.py` constructs the Hybrid Carrier prompts.
- `scripts/`: shell entry points for the workflow.

## Environment

The pipeline requires a Linux NVIDIA GPU with enough memory for FLUX and the selected evaluation models. Create the environment with `conda env create -f environment.yml && conda activate carrier-repro`. Install the official Diffusers repository and set `FLUX_LORA_TRAINER` to `examples/dreambooth/train_dreambooth_lora_flux.py`. Configure FLUX, SAM3 and Qwen2.5-VL in `configs/subject.env`. Set `DINO_MODEL` to run the optional DINOv3/SAM3 subject-preservation evaluation.

## Reproduce one white-box run

Copy `configs/subject.env.example` to `configs/subject.env` and fill in the model paths, one reference directory, the subject and ImageNet source/target classes, and output paths. Run these commands from the repository root:

1. Train a subject LoRA from its included reference photographs: `bash scripts/01_train_lora.sh configs/subject.env`. Initially `EXECUTE=0` prints/checks the command; set `EXECUTE=1` to launch training.
2. Generate 200 LoRA subject-sample candidates: `bash scripts/02_generate_subject_samples.sh configs/subject.env`. Inspect the images and choose one with a clear, single subject and minimal duplication/ghosting; set `SOURCE_IMAGE` to that image.
3. Set `CONSTRUCTION_CONDITION` to `target_carrier`, `non_target_carrier`, or `hybrid_carrier`, then run `bash scripts/03_run_carrier_attacks.sh configs/subject.env`. The value is passed to `flux_automation.py --construction-condition`; that automation selects and constructs the requested carrier. Under the selected condition it runs CRA, JIA and CIRA, writing per-method outputs, Grad-CAM overlays, `summary.json`, and a contact image under `OUTPUT_ROOT/RUN_NAME_<condition>/`. If `DINO_MODEL` is set, it also writes DINOv3/SAM3 subject-preservation CSV/JSON. Set `EXECUTE=0` to inspect the commands first.

Hybrid Carrier uses the included target-specific prompt catalog for its 30 listed target classes. For another ImageNet target, fill `HYBRID_FEATURE_CLAUSE`, `HYBRID_IDENTITY_CLAUSE` and `HYBRID_FORBIDDEN_CLAUSE` in the config; the same Non-Target Carrier class is used, with localized target-related attributes while retaining carrier identity.

`bash scripts/run_all.sh configs/subject.env` chains these stages. It stops after generating candidates until a valid `SOURCE_IMAGE` is selected; image selection is intentionally a manual quality-control step. For a subsequent attack-only run, use step 3 directly rather than retraining the LoRA.
