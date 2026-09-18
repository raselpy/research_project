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
import hashlib
import os
import time
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


def _atomic_torch_save(obj, final_path: Path, retries: int = 5, delay: float = 0.5) -> None:
    """Writes obj to a process-unique temp file then atomically renames
    it onto final_path, so a crash mid-write never leaves a truncated
    checkpoint at final_path (torch.save alone is not atomic).

    The rename is retried on PermissionError: on Windows, MoveFileEx
    can't overwrite a target that another process briefly has open
    (antivirus scanning, OneDrive sync, PyCharm's indexer/watcher can
    all do this), which raises WinError 5 even though the file is
    about to be free again. In train() we no longer call this
    repeatedly against the *same* final_path every epoch (that pattern
    is what invited another process to keep grabbing a handle on it) —
    each epoch now gets its own uniquely-named resume file, so this
    retry is a safety net for a first-time write, not the primary
    fix for the lock contention."""
    tmp_path = final_path.with_suffix(final_path.suffix + f".{os.getpid()}.tmp")
    torch.save(obj, tmp_path)
    for attempt in range(retries):
        try:
            os.replace(tmp_path, final_path)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            time.sleep(delay)


def _config_fingerprint(cfg) -> str:
    """Hash of the config fields that must match for a resume checkpoint
    to be safe to load. Deliberately excludes `logging` (tracking-only,
    doesn't affect what gets trained) and `experiment_name` (a label).
    Guards against silently resuming a partially-trained fold under a
    DIFFERENT config than the one that started it — e.g. Ctrl+C'ing a
    run, changing training.learning_rate or training.augmentation_preset
    (neither of which changes the model's tensor shapes, so
    load_state_dict wouldn't catch it), and re-running the same
    experiment_name/fold. Without this check that would silently
    resume the old partial-training state under the new hyperparameters
    with no warning at all."""
    from omegaconf import OmegaConf

    relevant = OmegaConf.masked_copy(cfg, ["dataset", "model", "training", "seed"])
    canonical = OmegaConf.to_yaml(relevant, resolve=True)
    return hashlib.sha256(canonical.encode()).hexdigest()


def load_case_folds(manifest_path: Path) -> dict[str, int]:
    """Reads data/processed/manifest.csv, returns {case_id: fold} for
    train_pool cases only. Raises a clear error if the manifest predates
    Phase 8's fold assignment (no `fold` column) rather than silently
    falling back to some other split — that would make it easy to
    accidentally train on a non-deterministic, non-partitioning split
    without noticing.

    Holdout cases (fold == -1, see prepare.py's assign_holdout_split)
    are excluded here — they must never appear in any fold's train or
    val set, since they exist specifically to be evaluated by the
    5-fold ensemble without any of the 5 models having trained on them
    (Phase 9's Table 2)."""
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "fold" not in reader.fieldnames:
            raise RuntimeError(
                f"{manifest_path} has no 'fold' column — re-run "
                f"`python -m src.datasets.prepare --path <raw data dir>` "
                f"(Phase 8 added deterministic k-fold assignment to prepare.py)."
            )
        return {row["case_id"]: int(row["fold"]) for row in reader if int(row["fold"]) >= 0}


def load_holdout_case_ids(manifest_path: Path) -> list[str]:
    """Reads data/processed/manifest.csv, returns the case IDs reserved
    as the genuine holdout set (prepare.py's assign_holdout_split) —
    the cases used for Phase 9's ensembled evaluation (Table 2), since
    none of the 5 fold models ever trained on them."""
    with open(manifest_path, newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or "split" not in reader.fieldnames:
            raise RuntimeError(
                f"{manifest_path} has no 'split' column — re-run "
                f"`python -m src.datasets.prepare --path <raw data dir>` "
                f"(adds the holdout/train_pool split alongside fold assignment)."
            )
        return [row["case_id"] for row in reader if row["split"] == "holdout"]


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
        self.scaler = torch.amp.GradScaler("cuda", enabled=(self.device.type == "cuda"))
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

    def _find_latest_resume_checkpoint(self, checkpoint_dir: Path) -> Path | None:
        """Resume checkpoints are now written one-per-epoch as
        fold_<i>_resume_epoch<N>.pt rather than repeatedly overwriting a
        single fixed filename. Overwriting one fixed path every epoch was
        what caused WinError 5 (PermissionError) on Windows: some other
        process (antivirus, OneDrive, PyCharm's indexer/watcher) would
        transiently hold a handle on that exact path right after it was
        written, and os.replace can't rename onto a path that's locked.
        Writing a fresh, uniquely-named file each epoch sidesteps that —
        nothing is ever asked to replace a file another process might
        currently have open. We just need to find the highest-epoch one
        to resume from."""
        pattern = f"fold_{self.cfg.dataset.fold}_resume_epoch*.pt"
        candidates = list(checkpoint_dir.glob(pattern))
        if not candidates:
            return None
        return max(candidates, key=lambda p: int(p.stem.rsplit("epoch", 1)[-1]))

    def _prune_old_resume_checkpoints(self, checkpoint_dir: Path, keep: Path | None) -> None:
        """Deletes every resume checkpoint for this fold except `keep`
        (or all of them, if keep is None e.g. after the fold finishes).
        Retries on PermissionError for the same Windows file-locking
        reason _atomic_torch_save does (antivirus/OneDrive/IDE indexer
        can transiently hold a handle right after a file is written)."""
        pattern = f"fold_{self.cfg.dataset.fold}_resume_epoch*.pt"
        for old in checkpoint_dir.glob(pattern):
            if old == keep:
                continue
            for attempt in range(5):
                try:
                    old.unlink(missing_ok=True)
                    break
                except PermissionError:
                    if attempt == 4:
                        logger.warning(f"Could not delete stale checkpoint {old} (file locked) — leaving it in place.")
                    else:
                        time.sleep(0.5)

    def validate(self, epoch: int) -> None:
        """Runs the held-out fold's val_case_ids through the model in
        eval mode (no grad, no augmentation) and logs the mean loss as
        val_loss. Reuses self.loss_fn directly, so it's exactly the same
        Dice+CE/Dice+BCE (with or without deep supervision) that trains
        the model — a monitoring signal, not a separate metric."""
        if not self.val_case_ids:
            return

        self.model.eval()
        val_losses = []
        with torch.no_grad():
            for _ in range(self.cfg.training.iterations_per_epoch):
                data, seg = load_batch(
                    self.val_case_ids,
                    self.processed_dir,
                    tuple(self.cfg.model.patch_size),
                    self.cfg.training.batch_size,
                    None,  # no augmentation at validation time
                    self.rng,
                )
                data, seg = data.to(self.device), seg.to(self.device)
                with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                    preds = self.model(data)
                    loss = self.loss_fn(preds, seg) if self.deep_supervision else self.loss_fn(preds[0], seg)
                val_losses.append(loss.item())

        mean_val_loss = float(np.mean(val_losses))
        logger.info(f"epoch {epoch}: val_loss={mean_val_loss:.4f}")
        log_epoch_metrics({"val_loss": mean_val_loss}, step=epoch)

    def train(self) -> None:
        # Phase 8: one subdirectory per experiment, one checkpoint file
        # per fold — cv.py's run_all_folds() launches 5 of these,
        # producing results/checkpoints/<experiment_name>/fold_<i>.pt.
        checkpoint_dir = Path("results/checkpoints") / str(self.cfg.experiment_name)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        final_checkpoint_path = checkpoint_dir / f"fold_{self.cfg.dataset.fold}.pt"

        # If a power cut (or any crash) interrupted this fold mid-training,
        # pick up from the last epoch we finished instead of epoch 0.
        start_epoch = 0
        latest_resume_path = self._find_latest_resume_checkpoint(checkpoint_dir)

        if latest_resume_path is not None:
            ckpt = torch.load(latest_resume_path, map_location=self.device)

            current_fingerprint = _config_fingerprint(self.cfg)
            saved_fingerprint = ckpt.get("config_fingerprint")
            if saved_fingerprint != current_fingerprint:
                raise ValueError(
                    f"{latest_resume_path} was saved under a different config than the one "
                    f"this run was started with (dataset/model/training/seed differ). Resuming "
                    f"would silently mix hyperparameters between the old partial run and this "
                    f"one. Either revert the config change, or delete this fold's stale resume "
                    f"checkpoints under {latest_resume_path.parent} and start the fold over."
                )

            self.model.load_state_dict(ckpt["model_state_dict"])
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])
            self.scaler.load_state_dict(ckpt["scaler_state_dict"])
            start_epoch = ckpt["epoch"] + 1
            logger.info(
                f"Resuming fold {self.cfg.dataset.fold} from epoch {start_epoch} " f"(found {latest_resume_path})"
            )

        with tracked_run(self.cfg, run_name=self.cfg.experiment_name):
            for epoch in range(start_epoch, self.cfg.training.epochs):
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
                    with torch.autocast(device_type=self.device.type, enabled=(self.device.type == "cuda")):
                        preds = self.model(data)
                        loss = self.loss_fn(preds, seg) if self.deep_supervision else self.loss_fn(preds[0], seg)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                    epoch_losses.append(loss.item())

                mean_loss = float(np.mean(epoch_losses))
                logger.info(f"epoch {epoch}: lr={lr:.5f}, train_loss={mean_loss:.4f}")
                log_epoch_metrics({"train_loss": mean_loss, "lr": lr}, step=epoch)

                if (epoch + 1) % self.cfg.training.val_every_n_epochs == 0:
                    self.validate(epoch)
                    self.model.train()  # switch back to train mode after eval

                # Save a resumable checkpoint after every epoch, so a power
                # cut only costs the in-progress epoch, not the whole fold.
                # Each epoch gets its own uniquely-numbered file (see
                # _find_latest_resume_checkpoint's docstring for why), and
                # older resume files for this fold are pruned only after
                # the new one is confirmed written.
                resume_checkpoint_path = checkpoint_dir / f"fold_{self.cfg.dataset.fold}_resume_epoch{epoch}.pt"
                _atomic_torch_save(
                    {
                        "epoch": epoch,
                        "model_state_dict": self.model.state_dict(),
                        "optimizer_state_dict": self.optimizer.state_dict(),
                        "scaler_state_dict": self.scaler.state_dict(),
                        "config_fingerprint": _config_fingerprint(self.cfg),
                    },
                    resume_checkpoint_path,
                )
                self._prune_old_resume_checkpoints(checkpoint_dir, keep=resume_checkpoint_path)

            _atomic_torch_save(self.model.state_dict(), final_checkpoint_path)
            logger.info(f"Saved checkpoint to {final_checkpoint_path}")
            # Fold finished cleanly - the resume files' job is done. Remove
            # them so a later re-run (e.g. different epoch count) doesn't
            # wrongly think this fold is already fully trained.
            self._prune_old_resume_checkpoints(checkpoint_dir, keep=None)


def main() -> None:
    # @hydra.main's config_path is resolved relative to *this file's
    # location on disk* by default — which breaks the moment this file
    # is installed non-editably (e.g. `pip install .`, as the
    # Dockerfile does), since it then physically lives inside
    # site-packages, nowhere near the real configs/ directory. Confirmed
    # as a real bug via an actual `docker compose run`: worked in every
    # local/editable-install test throughout this build, broke instantly
    # the first time run.py executed from an installed (non-editable)
    # package with "Primary config module 'configs' not found."
    #
    # Fixed by resolving the config directory from the current working
    # directory instead. Correct both for `python -m src.training.run`
    # (cwd = repo root locally) and for the installed `train`
    # console-script running inside Docker (WORKDIR /app, configs/
    # copied to /app/configs by the Dockerfile, or volume-mounted there
    # by docker-compose.yaml).
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
