from hydra.core.config_store import ConfigStore
from omegaconf import MISSING
from pydantic.dataclasses import dataclass

from src.config_schema.dataset import dataset_schema
from src.config_schema.logging import logging_schema
from src.config_schema.model import model_schema
from src.config_schema.training import training_schema


@dataclass
class Config:
    experiment_name: str = "brats_nnunet_runs"
    seed: int = 42
    dataset: dataset_schema.DatasetConfig = MISSING
    model: model_schema.ModelConfig = MISSING
    training: training_schema.TrainingConfig = MISSING
    logging: logging_schema.LoggingConfig = MISSING


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(name="config", node=Config)
    dataset_schema.setup_config()
    model_schema.setup_config()
    training_schema.setup_config()
    logging_schema.setup_config()