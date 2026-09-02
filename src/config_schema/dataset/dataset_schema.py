from dataclasses import field

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
    path: str = MISSING  # set per-machine in configs/dataset/brats2024.yaml
    # `split_ratios` still carves out a local held-out "validation-like" set,
    # since we don't have access to BraTS's real online judge for the true
    # validation/test cases — see Phase 8's note on this limitation.
    split_ratios: list[float] = field(default_factory=lambda: [0.8, 0.1, 0.1])
    seed: int = 42
    num_workers: int = 4
    # --- added for Phase 8 (5-fold CV) ---
    num_folds: int = 5  # paper: 5-fold CV on the training pool
    fold: int = 0  # which fold is held out as this run's CV-val split;
    # Phase 8 trains one model per fold value (0..4) via
    # Hydra multirun: `-m dataset.fold=0,1,2,3,4`
    fold_seed: int = 42  # must match across all 5 fold runs so the folds
    # are a consistent, non-overlapping partition


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="dataset", name="brats2024_dataset_schema", node=Brats2024DatasetSchema)
