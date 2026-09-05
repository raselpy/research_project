"""Evaluation entrypoint: runs inference (single checkpoint or
sigmoid-averaged ensemble) against the local held-out validation split,
computes per-case Dice/HD95 for the 3 BraTS regions (whole tumor, tumor
core, enhancing tumor — Section 2.1), and writes
results/tables/eval_results.json.

GAP FILLED HERE: like src/training/run.py (Phase 6), this file didn't
exist anywhere in this from-scratch build before Phase 8, despite being
referenced by dvc.yaml (Phase 3) and this phase's own ensembling section.

SIMPLIFICATION, stated up front: inference here evaluates a single
center crop of each case at cfg.model.patch_size, not a full
sliding-window scan of the whole volume (which is what nnU-Net actually
does at real test time). This keeps the evaluation loop simple and fast
enough to smoke-test, but the resulting Dice/HD95 numbers should be
read as "the pipeline computes real, correctly-implemented metrics on
real predictions" — not as directly comparable to the paper's own
whole-volume inference numbers. Sliding-window inference would be the
natural next improvement if these numbers need to be paper-comparable
later.
"""

import argparse
import glob
import json
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.config_schema import setup_config
from src.evaluation.ensemble import ensemble_predict
from src.evaluation.metrics import dice_score, hausdorff95
from src.logging_utils.setup import get_logger
from src.training.run import _random_crop, load_case_folds

logger = get_logger(__name__)

setup_config()


def pred_to_regions(pred: torch.Tensor, region_based: bool) -> dict[str, np.ndarray]:
    """Converts a model's final-resolution output (already
    probability-valued — see src/models/nnunet3d.py's SegmentationHead)
    into binary whole/core/enhancing masks for metric computation."""
    pred_np = pred.detach().cpu().numpy()
    if region_based:
        # (1, 3, D, H, W) independent sigmoid probs, channel order [WT, TC, ET]
        return {
            "whole": pred_np[0, 0] > 0.5,
            "core": pred_np[0, 1] > 0.5,
            "enhancing": pred_np[0, 2] > 0.5,
        }
    # (1, 4, D, H, W) softmax over {background, NCR, ED, ET}
    cls = pred_np[0].argmax(axis=0)
    return {
        "whole": cls > 0,
        "core": (cls == 1) | (cls == 3),
        "enhancing": cls == 3,
    }


def seg_to_regions(seg: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "whole": seg > 0,
        "core": (seg == 1) | (seg == 3),
        "enhancing": seg == 3,
    }


def evaluate_case(pred_regions: dict, gt_regions: dict, voxel_spacing=(1.0, 1.0, 1.0)) -> dict:
    scores = {}
    for region in ("whole", "core", "enhancing"):
        scores[f"{region}_dice"] = dice_score(pred_regions[region], gt_regions[region])
        scores[f"{region}_hd95"] = hausdorff95(pred_regions[region], gt_regions[region], voxel_spacing)
    return scores


def load_case_crop(case_id: str, processed_dir: Path, patch_size: tuple, rng: np.random.Generator):
    case_dir = processed_dir / case_id
    image = np.load(case_dir / "image.npy")
    seg = np.load(case_dir / "seg.npy")
    image_crop = _random_crop(image, patch_size, rng)
    seg_crop = _random_crop(seg[np.newaxis, ...], patch_size, rng)[0]
    return image_crop, seg_crop


def run_evaluation(
    cfg: DictConfig,
    checkpoint_paths: list[str],
    device: torch.device,
) -> list[dict]:
    processed_dir = Path("data/processed")
    case_folds = load_case_folds(processed_dir / "manifest.csv")
    # Evaluate on every case that was SOME fold's held-out val set —
    # i.e. the full dataset, each case scored by the fold whose
    # checkpoint(s) held it out. For a single-checkpoint (non-ensemble)
    # eval, this is just that one fold's held-out cases.
    eval_case_ids = sorted(case_folds.keys())

    rng = np.random.default_rng(cfg.seed)
    results = []
    for case_id in eval_case_ids:
        image_crop, seg_crop = load_case_crop(case_id, processed_dir, tuple(cfg.model.patch_size), rng)
        batch = torch.from_numpy(image_crop).float().unsqueeze(0)

        if len(checkpoint_paths) > 1:
            pred = ensemble_predict(checkpoint_paths, cfg, batch, device)
        else:
            model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(device)
            state_dict = torch.load(Path(checkpoint_paths[0]), map_location=device, weights_only=True)
            model.load_state_dict(state_dict)
            model.eval()
            with torch.no_grad():
                pred = model(batch.to(device))[-1]

        pred_regions = pred_to_regions(pred, cfg.model.region_based_training)
        gt_regions = seg_to_regions(seg_crop)
        scores = evaluate_case(pred_regions, gt_regions)
        scores["case_id"] = case_id
        results.append(scores)
        logger.info(f"{case_id}: {scores}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--checkpoint", type=str, help="Single checkpoint path")
    group.add_argument("--ensemble", type=str, help="Glob pattern matching multiple checkpoints")
    parser.add_argument("--patch-size", type=int, nargs=3, default=None)
    args, overrides = parser.parse_known_args()

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path("configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        cfg = compose(config_name="config", overrides=overrides)

    if args.patch_size:
        cfg.model.patch_size = list(args.patch_size)

    checkpoint_paths = sorted(glob.glob(args.ensemble)) if args.ensemble else [args.checkpoint]
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints matched: {args.ensemble}")

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    results = run_evaluation(cfg, checkpoint_paths, device)

    out_dir = Path("results/tables")
    out_dir.mkdir(parents=True, exist_ok=True)
    # Namespaced by experiment_name — a fixed "eval_results.json" would
    # silently overwrite between experiment variants, breaking Phase 8's
    # own final check (one ranking entry per trained variant) and Phase
    # 9's multi-variant comparison before either got a chance to run.
    out_path = out_dir / f"eval_results_{cfg.experiment_name}.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    logger.info(f"Wrote {len(results)} case result(s) to {out_path}")


if __name__ == "__main__":
    main()
