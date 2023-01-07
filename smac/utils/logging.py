from __future__ import annotations

import logging
import logging.config
from pathlib import Path

import yaml

import smac

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


def setup_logging(level: int | Path | None = None) -> None:
    """Sets up the logging configuration for all modules.

    Parameters
    ----------
    level : int | Path | None, defaults to None
        An integer representing the logging level. An custom logging configuration can be used when passing a path.
    """
    if isinstance(level, Path):
        log_filename = level
    else:
        path = Path() / smac.__file__
        log_filename = path.parent / "logging.yml"

    with (log_filename).open("r") as stream:
        config = yaml.safe_load(stream)

    if isinstance(level, int):
        config["root"]["level"] = level
        config["handlers"]["console"]["level"] = level

    logging.config.dictConfig(config)


def get_logger(logger_name: str) -> logging.Logger:
    """Get the logger by name."""
    logger = logging.getLogger(logger_name)
    return logger
