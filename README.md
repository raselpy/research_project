# research_project

Reproduction of Isensee et al., *[nnU-Net for Brain Tumor Segmentation](https://arxiv.org/abs/2011.00848)* (BraTS 2020 winner), using **BraTS 2024 GLI** NIfTI data.

## What this repo actually does

- **Model:** `NNUNet3D` (`src/models/nnunet3d.py`) — 4 MRI channels, instance or batch norm, deep-supervision heads, softmax (4 classes) or sigmoid (3 overlapping regions).
- **Loss:** Dice+CE (standard) or Dice+BCE (region-based), sample-Dice or batch-Dice, wrapped over all decoder outputs (`src/training/losses.py`). Heads already emit probabilities; losses do **not** use logits.
- **Training:** one fold per invocation — SGD + poly LR, `batchgenerators` augmentation, MLflow (optional DagsHub), checkpoints under `results/checkpoints/<experiment>/fold_<i>.pt`.
- **Evaluation:** Dice / HD95 on whole tumor, tumor core, and enhancing tumor. Inference is a **single crop** at `model.patch_size` (random crop in `src/evaluation/run.py`; center crop in the qualitative figure), **not** sliding-window whole-volume inference.
- **Protocol:** 5-fold CV on the training pool; a local holdout from `prepare.py --holdout-ratio` stands in for the paper’s BraTS validation set (this repo has no access to the online judge).

Those Dice/HD95 numbers prove the pipeline; they are **not** paper-comparable whole-volume scores. The smoke schedule used by `run_experiments.sh` is `epochs=10`, `iterations_per_epoch=20` — not the paper’s 1000 × 250.

## Layout

```
configs/                 Hydra groups: dataset, model, training, logging, experiment
src/config_schema/       Pydantic/Hydra ConfigStore schemas
src/datasets/prepare.py  BraTS 2024 GLI → data/processed
src/models/nnunet3d.py
src/training/            run.py, cv.py, losses.py, augmentation.py
src/evaluation/          metrics, ensemble, ranking, tables, qualitative figure, volume viewer
src/tracking/            MLflow + DagsHub
tests/
results/checkpoints/     fold_*.pt
results/tables/          holdout eval JSON, table1/2 CSV, ranking
results/tables/cv_val/   per-fold eval JSON
dvc.yaml                 prepare_data → train → evaluate
run_experiments.sh       8 ablation variants × 5 folds
```

## Setup

Python ≥ 3.11.

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill DagsHub / MLflow secrets; never commit .env
```

Console script: `train` → `src.training.run:main` (same as `python -m src.training.run`). Hydra config is composed from **cwd** `configs/`, not from `@hydra.main` next to the installed module.

### Secrets (DagsHub)

`logging.use_dagshub=true` requires environment variables (see `src/tracking/dagshub_utils.py`):

- `DAGSHUB_USERNAME`
- `DAGSHUB_REPO`
- `DAGSHUB_TOKEN`

`run_experiments.sh` sources `.env` if present. Local MLflow works with `logging.use_dagshub=false` (default in `configs/logging/default.yaml`). Optional DVC remote: `./setup_dvc.sh`.

## Data

Expected raw layout (BraTS-2024-GLI names, not BraTS-2020 `_t1.nii.gz`):

```
<data>/
  <case_id>/
    <case_id>-t1n.nii    # T1
    <case_id>-t1c.nii    # T1c
    <case_id>-t2w.nii    # T2
    <case_id>-t2f.nii    # FLAIR
    <case_id>-seg.nii
```

Default path in `configs/dataset/brats2024.yaml` (and `dvc.yaml`): `data/raw/BraTS2024_small_dataset`.

Prepare (z-score nonzero voxels, remap labels `{0,1,2,3}`, map resection cavity `4` → background, write `data/processed/` + `manifest.csv` with holdout + 5-fold IDs):

```bash
python -m src.datasets.prepare --path data/raw/BraTS2024_small_dataset
# optional: --num-folds 5 --fold-seed 42 --holdout-ratio 0.1
```

Or: `dvc repro prepare_data`.

Inspect a case (no GPU/checkpoints):

```bash
python -m src.evaluation.visualize_volume --case <case_id>
python -m src.evaluation.visualize_volume --image path/to/volume.nii --seg path/to/seg.nii
```

## Config and experiments

Compose: `dataset` / `model` / `training` / `logging` / `experiment` from `configs/config.yaml`. Overrides are Hydra `key=value` (not argparse flags) on `train` / `src.training.run`.

Schema defaults (`src/config_schema/`): `batch_size=2`, `epochs=1000`, `iterations_per_epoch=250`, `loss_type=dice_ce`, `augmentation_preset=baseline`, `num_classes=4`, `norm_type=instance`, `region_based_training=false`, `patch_size=[128,128,128]`.

| Config (`experiment=`) | Paper-style label | Main overrides |
|------------------------|-------------------|----------------|
| `baseline` | BL | schema defaults (bs=2) |
| `baseline_bs5` | BL* | `batch_size=5` |
| `ablation_region` | BL*+R | region training, `num_classes=3`, `dice_bce`, bs=5 |
| `ablation_dataug` | BL*+R+DA | + `augmentation_preset=DA` |
| `ablation_dataug_star` | BL*+R+DA* | + `DA_star` (per-channel brightness) |
| `ablation_batchnorm` | BL*+R+DA+BN | + `norm_type=batch` |
| `ablation_batchdice` | BL*+R+DA+BD | + `batch_dice=true` |
| `ablation_dataug_star_bn` | BL*+R+DA*+BN | DA* + BatchNorm |

Every region-based config **must** set `model.num_classes=3` and `training.loss_type=dice_bce`.

Augmentation presets (`src/training/augmentation.py`): `baseline` (nnU-Net default DA), `DA` (elastic, per-axis scale, brightness, gamma), `DA_star` (DA with per-MRI-channel brightness).

## Training

One fold:

```bash
python -m src.training.run
python -m src.training.run experiment=baseline experiment_name=baseline dataset.fold=0 dataset.num_folds=5
train training.epochs=1 training.iterations_per_epoch=2
```

All five folds (sequential subprocesses):

```bash
python -m src.training.cv --experiment baseline --num-folds 5
python -m src.training.cv --experiment ablation_region --num-folds 5 training.epochs=10
```

Writes `results/checkpoints/<experiment_name>/fold_<i>.pt`. Extra Hydra overrides after the argparse flags are forwarded to `run.py`.

### Ablation sweep

```bash
./run_experiments.sh              # pending experiment+fold jobs
./run_experiments.sh --status
MAX_PARALLEL=1 ./run_experiments.sh
GPUS="0,1" MAX_PARALLEL=2 ./run_experiments.sh
./run_experiments.sh --reset ablation_region
./run_experiments.sh --reset ablation_region 2
./run_experiments.sh --reset all
```

Resume is **per finished fold**, not per epoch. A crash mid-fold restarts that fold; completed folds are skipped (markers in `.run_state/`). Logs: `logs/<experiment>-fold<n>.log`. The script always passes `logging.use_dagshub=true` plus the smoke epoch/iteration overrides above.

Docker (config from `/app`; data/results/configs/logs mounted):

```bash
docker compose run train training.epochs=1 training.iterations_per_epoch=2
```

## Evaluation and paper tables

### Per-fold CV val (Table 1 pool)

Scores **only** the cases that fold held out (`split=cv_val`). Default when using `--checkpoint`.

```bash
python -m src.evaluation.run --checkpoint results/checkpoints/baseline/fold_0.pt experiment=baseline experiment_name=baseline
```

Writes `results/tables/cv_val/eval_results_<experiment_name>_fold<i>.json`.

### Holdout ensemble (Table 2 / Table 3)

Cases never used in any CV fold. Default when using `--ensemble`.

```bash
python -m src.evaluation.run --ensemble "results/checkpoints/baseline/fold_*.pt" experiment=baseline experiment_name=baseline
python -m src.evaluation.run --ensemble "results/checkpoints/ablation_region/fold_*.pt" --split holdout experiment=ablation_region experiment_name=ablation_region
```

Writes `results/tables/eval_results_<experiment_name>.json`. Ensemble is sigmoid-averaged fold predictions (`src/evaluation/ensemble.py`). You can pass `--patch-size D H W`. Hydra overrides after the flags still apply.

### Tables 1–3

```bash
python -m src.evaluation.tables
python -m src.evaluation.ranking
```

- Table 1: mean Dice (%) Whole / Core / Enh / Mean, pooled 5-fold CV-val → `results/tables/table1.csv`
- Table 2: holdout Dice + HD95 → `results/tables/table2.csv`
- Table 3: BraTS rank-then-aggregate (0 = best, 1 = worst) on holdout `eval_results_*.json` only (not `cv_val/`) → `results/tables/brats_ranking.json`

If those JSON files are missing, both modules run a toy self-test instead.

### Qualitative figure

Re-runs ensemble inference (center crop) for best / p75 / median / p25 / worst holdout cases by whole-tumor Dice. Overlay: NCR turquoise, ED violet, ET yellow.

```bash
python -m src.evaluation.qualitative_figure --variant ablation_dataug_star_bn
```

Needs `results/checkpoints/<variant>/fold_*.pt` and `results/tables/eval_results_<variant>.json`. Writes `results/tables/qualitative_<variant>.png`.

## DVC

`dvc.yaml` stages:

1. `prepare_data` — `python -m src.datasets.prepare --path data/raw/BraTS2024_small_dataset`
2. `train` — `python -m src.training.run` → `results/checkpoints`
3. `evaluate` — `python -m src.evaluation.run --checkpoint results/checkpoints/latest.pt` → `results/tables/eval_results.json`

Keep the prepare path in sync with `configs/dataset/brats2024.yaml` (DVC does not interpolate Hydra). Fold-aware eval uses `src.evaluation.run` as above, not the single `latest.pt` DVC evaluate command.

## Tests and CI

```bash
ruff check src/ tests/
black --check src/ tests/
mypy src
pytest --cov=src
```

GitHub Actions (`.github/workflows/ci.yaml`) installs **CPU** torch first, then `.[dev]`, then the same four checks.

## Paper map

| Paper | This repo |
|-------|-----------|
| §2.1 regions + rank-then-aggregate | `src/evaluation/metrics.py`, `ranking.py` |
| §2.2 SGD + poly LR, 3D U-Net, z-score | `src/training/run.py`, `nnunet3d.py`, `prepare.py` |
| §2.3 region-based training | `configs/experiment/ablation_region.yaml`, Dice+BCE |
| §2.4 DA / DA* / BN / BD | corresponding `ablation_*.yaml` |
| §3.1 Tables 1–2 | `src/evaluation/tables.py` |
| ranking table | `src/evaluation/ranking.py` |
