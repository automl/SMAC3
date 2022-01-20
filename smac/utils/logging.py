import logging
from typing import Union, List, Dict, Any

import numpy as np

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class PickableLoggerAdapter(object):
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
        return {'name': self.name}

    def __setstate__(self, state: Dict[str, Any]) -> None:
        """
        Method is called when pickle loads an object. Retrieves the name and
        creates a logger.
        Parameters
        ----------
        state - dictionary, containing the logger name.
        """
        self.name = state['name']
        self.logger = logging.getLogger(self.name)

    def debug(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.debug(msg, *args, **kwargs)

    def info(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.info(msg, *args, **kwargs)

    def warning(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.warning(msg, *args, **kwargs)

    def error(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.error(msg, *args, **kwargs)

    def exception(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.exception(msg, *args, **kwargs)

    def critical(self, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.critical(msg, *args, **kwargs)

    def log(self, level, msg, *args, **kwargs):  # type: ignore[no-untyped-def] # noqa F821
        self.logger.log(level, msg, *args, **kwargs)

    def isEnabledFor(self, level):  # type: ignore[no-untyped-def] # noqa F821
        return self.logger.isEnabledFor(level)


def format_array(input: Union[str, int, float, np.ndarray, list]) -> Union[str, List[str]]:
    """
    Transform a numpy array to a list of format so that it can be printed by logger.
    If the list holds one element only, then a string is returned.

    Parameters
    ----------
        input_array: np.ndarray or list.

    Returns
    -------
        result: str or list.
    """

    if isinstance(input, str) or isinstance(input, float) or isinstance(input, int):
        return str(input)

    if isinstance(input, np.ndarray):
        input = input.tolist()

    formatted_list = []
    for item in input:
        # https://stackoverflow.com/a/33482726
        formatted_list.append(f"{item:4f}")

    if len(input) == 1:
        return input[0]

    return formatted_list
