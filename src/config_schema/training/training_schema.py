from typing import Optional

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class TrainingConfig:
    _target_: str = MISSING


@dataclass
class NnunetBaselineTrainingSchema(TrainingConfig):
    _target_: str = "src.training.run.Trainer"
    epochs: int = 1000
    iterations_per_epoch: int = 250
    batch_size: int = 2
    learning_rate: float = 0.01
    momentum: float = 0.99
    optimizer: str = "sgd"
    lr_schedule: str = "poly"
    lr_poly_exponent: float = 0.9
    loss_type: str = "dice_ce"
    augmentation_preset: str = "baseline"  # "baseline" | "DA" | "DA_star"
    postprocess_enhancing_threshold: Optional[int] = None
    val_every_n_epochs: int = 50
    checkpoint_every_n_epochs: int = 100
    device: str = "cuda"


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(
        group="training",
        name="nnunet_baseline_training_schema",
        node=NnunetBaselineTrainingSchema,
    )
    