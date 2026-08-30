"""Console + rotating file logging setup. No paper-specific content —
used identically across every phase's scripts (prepare.py, run.py,
evaluation/run.py, ...).

Idempotent: calling setup_logging() more than once (e.g. once per module
that imports it) does not add duplicate handlers to the same logger.
"""
import logging
import logging.handlers
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler
    _HAS_RICH = True
except ImportError:
    _HAS_RICH = False


DEFAULT_LOGGER_NAME = "research_project"


def setup_logging(
    name: str = DEFAULT_LOGGER_NAME,
    level: str = "INFO",
    log_dir: str = "logs",
    log_filename: str = "run.log",
    max_bytes: int = 5 * 1024 * 1024,
    backup_count: int = 3,
) -> logging.Logger:
    """Configures and returns a logger with a console handler (Rich if
    available, else plain StreamHandler) and a rotating file handler.

    Safe to call repeatedly — checks `logger.handlers` first and returns
    the existing logger unchanged if it's already set up, rather than
    stacking duplicate handlers that would print/write each line twice.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    if _HAS_RICH:
        console_handler: logging.Handler = RichHandler(rich_tracebacks=True, show_path=False)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
    console_handler.setLevel(level)
    logger.addHandler(console_handler)

    log_dir_path = Path(log_dir)
    log_dir_path.mkdir(parents=True, exist_ok=True)
    file_handler = logging.handlers.RotatingFileHandler(
        log_dir_path / log_filename,
        maxBytes=max_bytes,
        backupCount=backup_count,
    )
    file_handler.setLevel(level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def get_logger(name: str = DEFAULT_LOGGER_NAME) -> logging.Logger:
    """Fetches the already-configured logger by name. If setup_logging()
    hasn't been called yet for this name, configures it with defaults
    first — so any module can just call get_logger() without worrying
    about import order relative to setup_logging()."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        return setup_logging(name=name)
    return logger


if __name__ == "__main__":
    logger = setup_logging()
    logger.info("Logging configured.")

    handlers_before = len(logger.handlers)
    setup_logging()
    handlers_after = len(logger.handlers)
    assert handlers_before == handlers_after, (
        f"setup_logging() is not idempotent: {handlers_before} handlers "
        f"before, {handlers_after} after a second call"
    )
    logger.info(f"Idempotency check passed ({handlers_after} handlers, no duplicates).")

    print("OK — see logs/run.log")