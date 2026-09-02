"""DagsHub integration: redirects MLflow's tracking URI to a DagsHub
remote. Called automatically by src.tracking.mlflow_utils.init_mlflow()
whenever cfg.logging.use_dagshub=True.

Credentials (DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN) are read from
the environment, never from the Hydra config — secrets don't belong in
config files that get committed to Git (see .env.example from Phase 0).
Uses dagshub.auth.add_app_token() for non-interactive token auth, so this
never triggers a browser-based OAuth popup (which would hang in CI or on
a headless machine).
"""

import os

from src.logging_utils.setup import get_logger

logger = get_logger(__name__)


def init_dagshub(cfg) -> None:
    """No-op if cfg.logging.use_dagshub is falsy. Otherwise reads
    DagsHub credentials from the environment and redirects MLflow's
    tracking URI to the DagsHub remote for cfg.logging.mlflow_experiment_name
    to be tracked against."""
    use_dagshub = getattr(getattr(cfg, "logging", cfg), "use_dagshub", False)
    if not use_dagshub:
        return

    username = os.environ.get("DAGSHUB_USERNAME")
    repo = os.environ.get("DAGSHUB_REPO")
    token = os.environ.get("DAGSHUB_TOKEN")

    missing = [
        name
        for name, val in [
            ("DAGSHUB_USERNAME", username),
            ("DAGSHUB_REPO", repo),
            ("DAGSHUB_TOKEN", token),
        ]
        if not val
    ]
    if missing:
        raise RuntimeError(
            f"cfg.logging.use_dagshub=True but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not set in the environment "
            f"(see .env.example)."
        )
    # mypy can't trace non-None-ness through the `missing` list
    # comprehension above, even though the raise on the line before
    # guarantees all three are set by this point — narrow explicitly.
    assert username is not None and repo is not None and token is not None

    import dagshub
    import dagshub.auth

    dagshub.auth.add_app_token(token)
    dagshub.init(repo_name=repo, repo_owner=username, mlflow=True)

    import mlflow

    logger.info(f"MLflow tracking URI now points at DagsHub: {mlflow.get_tracking_uri()}")


if __name__ == "__main__":
    from omegaconf import OmegaConf

    cfg = OmegaConf.create({"logging": {"use_dagshub": True}})
    init_dagshub(cfg)
    print("OK — MLflow tracking URI redirected to DagsHub.")
