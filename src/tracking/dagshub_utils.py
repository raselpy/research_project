"""DagsHub integration: redirects MLflow's tracking URI to a DagsHub
remote. Called automatically by src.tracking.mlflow_utils.init_mlflow()
whenever cfg.logging.use_dagshub=True.

Credentials (DAGSHUB_USERNAME, DAGSHUB_REPO, DAGSHUB_TOKEN) are read from
the environment, never from the Hydra config — secrets don't belong in
config files that get committed to Git. Uses dagshub.auth.add_app_token()
for non-interactive token auth, so this never triggers a browser-based
OAuth popup.
"""
import os

from src.logging_utils.setup import get_logger

logger = get_logger(__name__)


def init_dagshub(cfg) -> None:
    use_dagshub = getattr(getattr(cfg, "logging", cfg), "use_dagshub", False)
    if not use_dagshub:
        return

    username = os.environ.get("DAGSHUB_USERNAME")
    repo = os.environ.get("DAGSHUB_REPO")
    token = os.environ.get("DAGSHUB_TOKEN")

    missing = [name for name, val in [("DAGSHUB_USERNAME", username), ("DAGSHUB_REPO", repo), ("DAGSHUB_TOKEN", token)] if not val]
    if missing:
        raise RuntimeError(
            f"cfg.logging.use_dagshub=True but {', '.join(missing)} "
            f"{'is' if len(missing) == 1 else 'are'} not set in the environment "
            f"(see .env.example)."
        )

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