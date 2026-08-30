from hydra.core.config_store import ConfigStore
from pydantic.dataclasses import dataclass


@dataclass
class LoggingConfig:
    level: str = "INFO"
    log_dir: str = "logs"
    mlflow_experiment_name: str = "brats_nnunet"
    use_dagshub: bool = True


def setup_config() -> None:
    cs = ConfigStore.instance()
    cs.store(group="logging", name="default_logging_schema", node=LoggingConfig)