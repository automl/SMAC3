import logging
import logging.config
from pathlib import Path
import numpy as np
import os
from typing import Union, List, Iterable

import yaml

import smac

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


path = Path() / smac.__file__
with (path.parent / "logging.yml").open("r") as stream:
    config = yaml.load(stream, Loader=yaml.FullLoader)


class CustomFormatter(logging.Formatter):
    def format(self, record):
        # We cut the pathname
        if "pathname" in record.__dict__.keys():
            # truncate the pathname
            filename = os.path.basename(record.pathname)
            if len(filename) > 20:
                filename = "{}~{}".format(filename[:3], filename[-16:])
            record.pathname = filename

        return super(CustomFormatter, self).format(record)


logging.config.dictConfig(config)


def get_logger(logger_name: str) -> logging.Logger:
    logger = logging.getLogger(logger_name)

    # TODO: Fix
    # logger.propagate = False
    # logger_handler = logging.StreamHandler()
    # logger_handler.setFormatter(CustomFormatter())
    # logger.handlers.clear()
    # logger.addHandler(logger_handler)

    return logger


'''
class PickableLoggerAdapter:
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __getstate__(self) -> Dict[str, str]:
        """
        Method is called when pickle dumps an object.

        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {"name": self.name}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.

        Parameters
        ----------
        state - dictionary, containing the logger name.
        """
        self.name = state["name"]
        self.logger = logging.getLogger(self.name)

    def debug(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Debug method."""
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Info method."""
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Warning method."""
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Error method."""
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Exception method."""
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Critical method."""
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        """Log method."""
        self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level):  # type: ignore[no-untyped-def] # noqa F821
        """Check if logger is enabled for a given level."""
        return self.logger.isEnabledFor(level)
'''


# TODO: Move me
def format_array(
    inputs: Union[str, int, float, np.ndarray, list], format_vals: bool = True
) -> Union[float, List[float]]:
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
