# Carrier-Based Personalized Adversarial Generation

This repository provides the implementation used to reproduce subject-specific FLUX LoRA training, subject-sample generation, carrier construction, CRA, JIA, CIRA, No-Carrier experiments, transfer evaluation, subject-preservation evaluation, and result aggregation.

The repository contains code and experiment configuration files. Personalized reference images, selected subject samples, trained LoRA weights, foundation-model checkpoints, and previously generated experiment outputs are not redistributed. The scripts generate their own runtime images, per-step CSV files, evaluation JSON files, transfer results, Grad-CAM visualizations, manifests, and final summary tables.

## 1. Environment

```bash
conda env create -f environment.yml
conda activate carrier-repro
accelerate config
```

The pipeline additionally requires FLUX.1-Kontext-dev, the official Diffusers `train_dreambooth_lora_flux.py` trainer, SAM3, Qwen2.5-VL-7B-Instruct, DINOv3 ViT-L/16, and torchvision ImageNet classifiers. Set local paths in `configs/subject.env`; the supplied example contains placeholders only.

## 2. Train the subject-specific LoRAs

Prepare the provided DreamBooth reference images locally for each of the 20 subjects. For each subject, set its subject name, `[V] subject` instance prompt, reference directory, output directory, and model paths, then run:

```bash
bash scripts/01_train_lora.sh configs/subject.env
```

The launcher uses the official Diffusers FLUX DreamBooth LoRA trainer. It prints the full command first; set `EXECUTE=1` to start training.

## 3. Generate 200 subject samples

For each trained LoRA:

```bash
bash scripts/02_generate_subject_samples.sh configs/subject.env
```

The original generation program creates 200 unique prompt/seed combinations and writes generated images, prompt text files, and `metadata.csv`. Select one clean sample with a single dominant subject, minimal ghosting, no duplicated body parts, and little distracting content. We refer to the selected clean generation as a **subject sample**.

`code/lora/select_subject_samples.py` retains the original SAM3-based ranking and contact-sheet workflow for optional automatic pre-screening.

## 4. Carrier construction and attacks

Set `SOURCE_IMAGE` to the selected subject sample and run:

```bash
bash scripts/03_run_carrier_attacks.sh configs/subject.env
```

The default method set is `cra,jia,cira`:

- **CRA**: carrier composite followed by the global finite-return classifier attack.
- **JIA**: joint mask-guided inpainting and classifier-guided optimization.
- **CIRA**: independent clean inpainting followed by the global finite-return classifier attack.

SAM3 masks are used for clean carrier construction and inpainting. The historical protected-mask output-directory creation is commented out in `flux_auomation.py`; the public orchestration creates CRA, JIA, and CIRA routes only.

Keep `EXECUTE=0` for command inspection. After checking the resolved paths and commands, set `EXECUTE=1`.

## 5. Formal batches and evaluation

The original formal experiment programs are retained under `code/formal_pipeline/`:

- `run_formal_17x30_shard.py`: Target Carrier, Non-Target Carrier, and Hybrid Carrier batches;
- `run_no_carrier_20x30_shard.py`: No-Carrier batches;
- `flux_auomation.py`: per-case construction, CRA/JIA/CIRA, Grad-CAM, and transfer evaluation;
- `evaluate_subject_preservation_batch.py`: DINOv3 and SAM3 preservation evaluation;
- `summarize_formal_1200.py`: final CSV/JSON aggregation;
- `audit_hybrid_600.py`: Hybrid Carrier completeness audit.

Update `code/formal_pipeline/dreambooth_20_sources.csv` with local subject-sample and LoRA paths before launching a formal shard:

```bash
cd code/formal_pipeline
python run_formal_17x30_shard.py \
  --num-shards 1 \
  --shard-index 0 \
  --construction-condition hybrid \
  --dry-run
```

Remove `--dry-run` only after all asset and model paths have been verified.

## 6. Convenience entry point

When the configuration already points to a trained LoRA and selected subject sample:

```bash
bash scripts/run_all.sh configs/subject.env
```

On a first run, subject selection remains a manual quality-control step between generation and attack execution.

## Repository layout

```text
code/
  lora/             # original LoRA launcher, 200-sample generator, SAM3 ranking
  formal_pipeline/  # original construction, attack, evaluation, aggregation
configs/            # local path template
scripts/            # shell entry points
environment.yml
```
