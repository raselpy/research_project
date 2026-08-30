from typing import List

from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass


@dataclass
class DatasetConfig:
    _target_: str = MISSING
    name: str = MISSING
    path: str = MISSING


@dataclass
class Brats2024DatasetSchema(DatasetConfig):
    _target_: str = "src.datasets.prepare.BraTS2024Dataset"
    name: str = "brats2024"
    path: str = MISSING
    split_ratios: List[float] = (0.8, 0.1, 0.1)
    seed: int = 42
    num_workers: int = 4
    num_folds: int = 5
    fold: int = 0
    fold_seed: int = 42


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="dataset", name="brats2024_dataset_schema", node=Brats2024DatasetSchema)