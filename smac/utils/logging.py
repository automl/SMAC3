from __future__ import annotations

from typing import Iterable

import logging
import logging.config
from pathlib import Path

import numpy as np
import yaml

import smac

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


def setup_logging(level: int | Path | None = None) -> None:
    """Sets up the logging configuration for all modules.

    Parameters
    ----------
    level : int | Path | None, defaults to None
        An integer representing the logging level. An own logging configuration can be used when passing a path.
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
    """Get the logger by name"""
    logger = logging.getLogger(logger_name)
    return logger


# TODO: Move me
def format_array(inputs: str | int | float | np.ndarray | list, format_vals: bool = True) -> float | list[float]:
    """Transform a numpy array to a list of format so that it can be printed by logger. If the list
    holds one element only, then a formatted string is returned.

    Parameters
    ----------
        inputs: np.ndarray or list.
            inputs value, could be anything serializable or a np array
        format_vals: bool.
            if the items in list are formatted values

    Returns
    -------
        result: float or list of floats.
    """
    if isinstance(inputs, np.ndarray):
        inputs = inputs.tolist()

    if not isinstance(inputs, Iterable):
        inputs = [inputs]

    formatted_list = []
    for item in inputs:
        item = float(item)
        if format_vals:
            item = np.round(item, 4)

        formatted_list.append(item)

    if len(formatted_list) == 1:
        return formatted_list[0]

    return formatted_list
