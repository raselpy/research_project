# """Training entrypoint. A single invocation trains one fold end-to-end.
# CLI convention: Hydra's key=value overrides (train training.epochs=1 ...),
# not argparse — matches Phase 8's cv.py, which calls this via subprocess
# with dataset.fold=0-style overrides.
# """
#
# import csv
# from pathlib import Path
#
# import hydra
# import numpy as np
# import torch
# from omegaconf import DictConfig
#
# from src.config_schema import setup_config
# from src.logging_utils.setup import get_logger
# from src.tracking.mlflow_utils import init_mlflow, log_epoch_metrics, tracked_run
# from src.training.augmentation import build_augmentation
# from src.training.losses import DeepSupervisionWrapper, build_loss
#
# logger = get_logger(__name__)
# setup_config()  # must register structured configs before @hydra.main composes
#
#
# def load_case_ids(manifest_path: Path) -> list[str]:
#     with open(manifest_path, newline="") as f:
#         reader = csv.DictReader(f)
#         return [row["case_id"] for row in reader]
#
#
# def _random_crop(volume: np.ndarray, patch_size: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
#     spatial_shape = volume.shape[1:]
#     starts = [int(rng.integers(0, max(dim - p + 1, 1))) for dim, p in zip(spatial_shape, patch_size)]
#     slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
#     cropped = volume[(slice(None),) + slices]
#     pad_widths = [(0, 0)] + [(0, max(p - c, 0)) for p, c in zip(patch_size, cropped.shape[1:])]
#     if any(w[1] > 0 for w in pad_widths):
#         cropped = np.pad(cropped, pad_widths, mode="constant")
#     return cropped
#
#
# def load_batch(case_ids, processed_dir, patch_size, batch_size, augmentation, rng):
#     chosen = rng.choice(case_ids, size=batch_size, replace=True)
#     images, segs = [], []
#     for case_id in chosen:
#         case_dir = processed_dir / case_id
#         image = np.load(case_dir / "image.npy")
#         seg = np.load(case_dir / "seg.npy")[np.newaxis, ...]
#         crop_seed = rng.integers(0, 2**31)
#         image_crop = _random_crop(image, patch_size, np.random.default_rng(crop_seed))
#         seg_crop = _random_crop(seg, patch_size, np.random.default_rng(crop_seed))  # same seed -> same crop location
#         images.append(image_crop)
#         segs.append(seg_crop)
#     batch = {"data": np.stack(images).astype(np.float32), "seg": np.stack(segs).astype(np.float32)}
#     if augmentation is not None:
#         batch = augmentation(**batch)
#     data = torch.from_numpy(batch["data"]).float()
#     seg = torch.from_numpy(batch["seg"]).long().squeeze(1)
#     return data, seg
#
#
# class Trainer:
#     def __init__(self, cfg: DictConfig):
#         self.cfg = cfg
#         self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
#         if cfg.training.device == "cuda" and self.device.type == "cpu":
#             logger.warning("cfg.training.device='cuda' but no GPU available — falling back to CPU.")
#
#         self.model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(self.device)
#         base_loss = build_loss(cfg.training, cfg.model)
#         self.deep_supervision = cfg.model.deep_supervision
#         self.loss_fn = DeepSupervisionWrapper(base_loss) if self.deep_supervision else base_loss
#
#         self.optimizer = torch.optim.SGD(
#             self.model.parameters(), lr=cfg.training.learning_rate, momentum=cfg.training.momentum, nesterov=True
#         )
#         self.augmentation = build_augmentation(cfg.training.augmentation_preset, patch_size=tuple(cfg.model.patch_size))
#
#         self.processed_dir = Path("data/processed")
#         all_case_ids = load_case_ids(self.processed_dir / "manifest.csv")
#         rng = np.random.default_rng(cfg.dataset.seed)
#         shuffled = rng.permutation(all_case_ids).tolist()
#         split_idx = max(int(len(all_case_ids) * cfg.dataset.split_ratios[0]), 1)
#         self.train_case_ids = shuffled[:split_idx]
#         self.val_case_ids = shuffled[split_idx:] or shuffled[:1]
#         logger.info(f"Train cases: {len(self.train_case_ids)}, val cases: {len(self.val_case_ids)}")
#         self.rng = np.random.default_rng(cfg.seed)
#
#     def _lr_at_epoch(self, epoch: int) -> float:
#         progress = epoch / max(self.cfg.training.epochs, 1)
#         return self.cfg.training.learning_rate * (1 - progress) ** self.cfg.training.lr_poly_exponent
#
#     def train(self) -> None:
#         checkpoint_dir = Path("results/checkpoints")
#         checkpoint_dir.mkdir(parents=True, exist_ok=True)
#
#         with tracked_run(self.cfg, run_name=self.cfg.experiment_name):
#             for epoch in range(self.cfg.training.epochs):
#                 lr = self._lr_at_epoch(epoch)
#                 for g in self.optimizer.param_groups:
#                     g["lr"] = lr
#
#                 self.model.train()
#                 epoch_losses = []
#                 for _ in range(self.cfg.training.iterations_per_epoch):
#                     data, seg = load_batch(
#                         self.train_case_ids,
#                         self.processed_dir,
#                         tuple(self.cfg.model.patch_size),
#                         self.cfg.training.batch_size,
#                         self.augmentation,
#                         self.rng,
#                     )
#                     data, seg = data.to(self.device), seg.to(self.device)
#                     self.optimizer.zero_grad()
#                     preds = self.model(data)
#                     loss = self.loss_fn(preds, seg) if self.deep_supervision else self.loss_fn(preds[0], seg)
#                     loss.backward()
#                     self.optimizer.step()
#                     epoch_losses.append(loss.item())
#
#                 mean_loss = float(np.mean(epoch_losses))
#                 logger.info(f"epoch {epoch}: lr={lr:.5f}, train_loss={mean_loss:.4f}")
#                 log_epoch_metrics({"train_loss": mean_loss, "lr": lr}, step=epoch)
#
#             checkpoint_path = checkpoint_dir / "latest.pt"
#             torch.save(self.model.state_dict(), checkpoint_path)
#             logger.info(f"Saved checkpoint to {checkpoint_path}")
#
#
# def main() -> None:
#     import sys
#
#     from hydra import compose, initialize_config_dir
#     from hydra.core.global_hydra import GlobalHydra
#
#     config_dir = str(Path("configs").resolve())
#     GlobalHydra.instance().clear()
#     with initialize_config_dir(version_base=None, config_dir=config_dir):
#         overrides = sys.argv[1:]
#         cfg = compose(config_name="config", overrides=overrides)
#         init_mlflow(cfg.logging)
#         Trainer(cfg).train()
#
#
# if __name__ == "__main__":
#     main()

"""Training entrypoint. A single invocation trains one fold end-to-end:
real data loading from data/processed/manifest.csv, real augmentation,
the real NNUNet3D model, the real Dice+CE/Dice+BCE loss with deep
supervision, SGD with the paper's poly LR schedule (Section 2.2),
MLflow-tracked metrics, and a checkpoint written to results/checkpoints/.

GAP FILLED HERE: the plan's Phase 3/6/8 sections refer to this file as a
"previously-tested version" from an earlier iteration of this project
that does not exist in this from-scratch build. Written here in Phase 6,
since Phase 6's Docker acceptance check is the first phase that actually
requires a working training entrypoint. Phase 8 extends this (adds the
fold-loop wrapper, src/training/cv.py) rather than replacing it.

CLI convention: uses Hydra's own `key=value` override syntax throughout
(e.g. `train training.epochs=1 training.iterations_per_epoch=2`), not
argparse `--flag value` — this needs to match Phase 8's `cv.py`, which
already calls this script with `dataset.fold=0`-style overrides via
subprocess. Docker's acceptance check command is written accordingly.

@hydra.main is NOT used, deliberately — see main()'s own docstring below
for why: its config_path resolves relative to this file's location on
disk, which breaks the moment this module is installed non-editably
(e.g. `pip install .`, as the Dockerfile does) and physically moves into
site-packages. Confirmed as a real deployment bug via an actual
`docker compose run`: worked in every local/editable-install test
throughout this build, broke instantly the first time run.py executed
from an installed (non-editable) package with "Primary config module
'configs' not found." Config composition instead happens manually in
main(), resolving the config directory from the current working
directory (correct both for local `python -m src.training.run` and for
the installed `train` console-script inside Docker).
"""

import csv
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

from src.config_schema import setup_config
from src.logging_utils.setup import get_logger
from src.tracking.mlflow_utils import init_mlflow, log_epoch_metrics, tracked_run
from src.training.augmentation import build_augmentation
from src.training.losses import DeepSupervisionWrapper, build_loss

logger = get_logger(__name__)

# Structured configs must be registered before @hydra.main composes the
# config on module load / first call — doing this inside main() would be
# too late, since cfg arrives already composed by then.
setup_config()


def load_case_folds(manifest_path: Path) -> dict[str, int]:
    """Reads data/processed/manifest.csv, returns {case_id: fold}. Raises
    a clear error if the manifest predates Phase 8's fold assignment
    (no `fold` column) rather than silently falling back to some other
    split — that would make it easy to accidentally train on a
    non-deterministic, non-partitioning split without noticing."""
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "fold" not in reader.fieldnames:
            raise RuntimeError(
                f"{manifest_path} has no 'fold' column — re-run "
                f"`python -m src.datasets.prepare --path <raw data dir>` "
                f"(Phase 8 added deterministic k-fold assignment to prepare.py)."
            )
        return {row["case_id"]: int(row["fold"]) for row in reader}


def _random_crop(volume: np.ndarray, patch_size: tuple[int, int, int], rng: np.random.Generator) -> np.ndarray:
    """Random spatial crop matching patch_size from a (C, H, W, D) volume.
    Pads with zeros if the source volume is smaller than patch_size in
    any dimension (relevant for small synthetic/test volumes)."""
    spatial_shape = volume.shape[1:]
    starts = [int(rng.integers(0, max(dim - p + 1, 1))) for dim, p in zip(spatial_shape, patch_size)]
    slices = tuple(slice(s, s + p) for s, p in zip(starts, patch_size))
    cropped = volume[(slice(None),) + slices]

    pad_widths = [(0, 0)] + [(0, max(p - c, 0)) for p, c in zip(patch_size, cropped.shape[1:])]
    if any(w[1] > 0 for w in pad_widths):
        cropped = np.pad(cropped, pad_widths, mode="constant")
    return cropped


def load_batch(
    case_ids: list[str],
    processed_dir: Path,
    patch_size: tuple[int, int, int],
    batch_size: int,
    augmentation,
    rng: np.random.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Loads batch_size cases (sampled with replacement from case_ids),
    random-crops each to patch_size, applies augmentation, returns
    (data, seg) tensors ready for the model."""
    chosen = rng.choice(case_ids, size=batch_size, replace=True)

    images, segs = [], []
    for case_id in chosen:
        case_dir = processed_dir / case_id
        image = np.load(case_dir / "image.npy")  # (4, H, W, D)
        seg = np.load(case_dir / "seg.npy")[np.newaxis, ...]  # (1, H, W, D)

        crop_seed = rng.integers(0, 2**31)
        image_crop = _random_crop(image, patch_size, np.random.default_rng(crop_seed))
        seg_crop = _random_crop(seg, patch_size, np.random.default_rng(crop_seed))  # same seed -> same crop location

        images.append(image_crop)
        segs.append(seg_crop)

    batch = {
        "data": np.stack(images).astype(np.float32),
        "seg": np.stack(segs).astype(np.float32),
    }
    if augmentation is not None:
        batch = augmentation(**batch)

    data = torch.from_numpy(batch["data"]).float()
    seg = torch.from_numpy(batch["seg"]).long().squeeze(1)  # (N, H, W, D)
    return data, seg


class Trainer:
    """Single-fold training loop: real model, real loss (with deep
    supervision), real augmentation, real data — SGD with the paper's
    poly LR schedule (Section 2.2)."""

    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
        if cfg.training.device == "cuda" and self.device.type == "cpu":
            logger.warning("cfg.training.device='cuda' but no GPU available — falling back to CPU.")

        self.model = hydra.utils.instantiate(cfg.model, _convert_="partial").to(self.device)
        base_loss = build_loss(cfg.training, cfg.model)
        self.deep_supervision = cfg.model.deep_supervision
        self.loss_fn = DeepSupervisionWrapper(base_loss) if self.deep_supervision else base_loss

        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg.training.learning_rate,
            momentum=cfg.training.momentum,
            nesterov=True,
        )
        self.augmentation = build_augmentation(cfg.training.augmentation_preset, patch_size=tuple(cfg.model.patch_size))

        self.processed_dir = Path("data/processed")
        manifest_path = self.processed_dir / "manifest.csv"
        case_folds = load_case_folds(manifest_path)

        this_fold = cfg.dataset.fold
        self.train_case_ids = [c for c, f in case_folds.items() if f != this_fold]
        self.val_case_ids = [c for c, f in case_folds.items() if f == this_fold]
        if not self.val_case_ids:
            raise ValueError(
                f"dataset.fold={this_fold} matches no cases in the manifest "
                f"(fold values present: {sorted(set(case_folds.values()))})."
            )
        logger.info(f"Fold {this_fold}: train cases: {len(self.train_case_ids)}, val cases: {len(self.val_case_ids)}")

        self.rng = np.random.default_rng(cfg.seed)

    def _lr_at_epoch(self, epoch: int) -> float:
        """Polynomial LR decay, per Section 2.2: lr * (1 - epoch/max_epochs)^exponent"""
        progress = epoch / max(self.cfg.training.epochs, 1)
        return self.cfg.training.learning_rate * (1 - progress) ** self.cfg.training.lr_poly_exponent

    def train(self) -> None:
        # Phase 8: one subdirectory per experiment, one checkpoint file
        # per fold — cv.py's run_all_folds() launches 5 of these,
        # producing results/checkpoints/<experiment_name>/fold_<i>.pt.
        checkpoint_dir = Path("results/checkpoints") / str(self.cfg.experiment_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        with tracked_run(self.cfg, run_name=self.cfg.experiment_name):
            for epoch in range(self.cfg.training.epochs):
                lr = self._lr_at_epoch(epoch)
                for g in self.optimizer.param_groups:
                    g["lr"] = lr

                self.model.train()
                epoch_losses = []
                for _ in range(self.cfg.training.iterations_per_epoch):
                    data, seg = load_batch(
                        self.train_case_ids,
                        self.processed_dir,
                        tuple(self.cfg.model.patch_size),
                        self.cfg.training.batch_size,
                        self.augmentation,
                        self.rng,
                    )
                    data, seg = data.to(self.device), seg.to(self.device)

                    self.optimizer.zero_grad()
                    preds = self.model(data)
                    loss = self.loss_fn(preds, seg) if self.deep_supervision else self.loss_fn(preds[0], seg)
                    loss.backward()
                    self.optimizer.step()
                    epoch_losses.append(loss.item())

                mean_loss = float(np.mean(epoch_losses))
                logger.info(f"epoch {epoch}: lr={lr:.5f}, train_loss={mean_loss:.4f}")
                log_epoch_metrics({"train_loss": mean_loss, "lr": lr}, step=epoch)

            checkpoint_path = checkpoint_dir / f"fold_{self.cfg.dataset.fold}.pt"
            torch.save(self.model.state_dict(), checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")


def main() -> None:
    import sys

    from hydra import compose, initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra

    config_dir = str(Path("configs").resolve())
    GlobalHydra.instance().clear()
    with initialize_config_dir(version_base=None, config_dir=config_dir):
        overrides = sys.argv[1:]
        cfg = compose(config_name="config", overrides=overrides)
        init_mlflow(cfg.logging)
        Trainer(cfg).train()


if __name__ == "__main__":
    main()
