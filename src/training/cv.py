"""Launches Phase 6's training entrypoint once per fold, producing one
checkpoint per fold under results/checkpoints/<experiment>/fold_<i>.pt.

Passes BOTH `experiment=<name>` (a Hydra defaults-group override,
selecting configs/experiment/<name>.yaml's actual config content — the
ablation variants Phase 9 adds) AND `experiment_name=<name>` (a plain
value override on the root Config.experiment_name field, which
Trainer.train() uses to name the checkpoint subdirectory). These look
redundant but serve different purposes — the first selects *what*
config to run, the second names *where its output goes* — and Hydra has
no built-in way to read back which defaults-group choice was made from
inside a manually-composed config (see src/training/run.py's docstring
on why this project doesn't use @hydra.main's automatic config
resolution), so this is the simplest correct way to keep them in sync.
"""

import argparse
import subprocess

from src.logging_utils.setup import get_logger

logger = get_logger(__name__)


def run_all_folds(experiment: str, num_folds: int = 5, extra_overrides: list[str] | None = None) -> None:
    """Runs `train` once per fold (0..num_folds-1) via subprocess, each
    producing results/checkpoints/<experiment>/fold_<i>.pt. Raises
    (via check=True) on the first fold that fails, rather than silently
    continuing with a partial set of checkpoints."""
    extra_overrides = extra_overrides or []
    for fold in range(num_folds):
        overrides = [
            f"experiment={experiment}",
            f"experiment_name={experiment}",
            f"dataset.fold={fold}",
            f"dataset.num_folds={num_folds}",
            *extra_overrides,
        ]
        logger.info(f"Fold {fold}/{num_folds - 1}: train {' '.join(overrides)}")
        subprocess.run(["train", *overrides], check=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment", type=str, required=True)
    parser.add_argument("--num-folds", type=int, default=5)
    args, unknown = parser.parse_known_args()
    run_all_folds(args.experiment, num_folds=args.num_folds, extra_overrides=unknown)
