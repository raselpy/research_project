from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra

from src.config_schema import setup_config


def test_config_composes_without_missing_experiment_group():
    """Regression test for the Phase 1 bug: configs/config.yaml's
    defaults list references `experiment: baseline`, but
    configs/experiment/baseline.yaml didn't exist anywhere in the
    original plan and compose() failed outright with
    MissingConfigException. Also covers the @package _global_ gap —
    without it, Hydra tries to nest the file's content under a
    non-existent `experiment:` key in the Config dataclass."""
    GlobalHydra.instance().clear()
    setup_config()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    assert cfg.model.name == "nnunet3d"
    assert cfg.dataset.name == "brats2024"
    assert cfg.training.learning_rate == 0.01


def test_num_classes_default_is_four_not_three():
    """Regression test for the Phase 6 bug, at the config-composition
    level rather than just the schema-default level (test_model.py
    covers that separately) — catches it even if someone overrides the
    default elsewhere in the config chain."""
    GlobalHydra.instance().clear()
    setup_config()
    with initialize(version_base=None, config_path="../configs"):
        cfg = compose(config_name="config")
    assert cfg.model.num_classes == 4
