"""MLflow experiment tracking plumbing. DagsHub integration (Phase 5)
routes through _init_dagshub_if_enabled() here — a no-op placeholder
until Phase 5 actually implements it.
"""
import dataclasses
from contextlib import contextmanager
from typing import Dict

import mlflow
from omegaconf import DictConfig, OmegaConf

from src.logging_utils.setup import get_logger


logger = get_logger(__name__)


def _init_dagshub_if_enabled(logging_cfg) -> None:
    if getattr(logging_cfg, "use_dagshub", False):
        logger.info("use_dagshub=True, but DagsHub integration isn't implemented until Phase 5 — tracking locally for now.")


def init_mlflow(logging_cfg) -> None:
    _init_dagshub_if_enabled(logging_cfg)
    experiment_name = getattr(logging_cfg, "mlflow_experiment_name", "brats_nnunet")
    mlflow.set_experiment(experiment_name)


def _flatten_cfg(cfg, parent_key: str = "", sep: str = ".") -> Dict[str, object]:
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        cfg = dataclasses.asdict(cfg)
    elif isinstance(cfg, DictConfig):
        cfg = OmegaConf.to_container(cfg, resolve=True)

    items: Dict[str, object] = {}
    if isinstance(cfg, dict):
        for k, v in cfg.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else str(k)
            items.update(_flatten_cfg(v, new_key, sep))
    else:
        items[parent_key] = cfg
    return items


@contextmanager
def tracked_run(cfg, run_name: str):
    with mlflow.start_run(run_name=run_name) as run:
        params = _flatten_cfg(cfg)
        mlflow.log_params({k: v for k, v in params.items() if v is not None})
        yield run


def log_epoch_metrics(metrics: Dict[str, float], step: int) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_artifact_dir(path: str) -> None:
    mlflow.log_artifacts(path)


if __name__ == "__main__":
    # from src.config_schema.logging.logging_schema import LoggingConfig
    from src.config_schema.logging.logging_schema import LoggingConfig

    logging_cfg = LoggingConfig(mlflow_experiment_name="brats_nnunet_smoke_test", use_dagshub=False)
    init_mlflow(logging_cfg)

    with tracked_run(logging_cfg, run_name="smoke_test") as run:
        logger.info(f"Started MLflow run: {run.info.run_id}")
        for epoch, value in enumerate([0.9, 0.7, 0.5]):
            log_epoch_metrics({"dummy_loss": value}, step=epoch)

    print("OK — smoke-test run logged. Run `mlflow ui` and check the 'brats_nnunet_smoke_test' experiment.")
