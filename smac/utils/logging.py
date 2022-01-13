import logging
import typing

import numpy as np

__copyright__ = "Copyright 2021, AutoML.org Freiburg-Hannover"
__license__ = "3-clause BSD"


class PickableLoggerAdapter(object):
    def __init__(self, name: str) -> None:
        self.name = name
        self.logger = logging.getLogger(self.name)

    def __getstate__(self) -> typing.Dict[str, str]:
        """
        Method is called when pickle dumps an object.
        Returns
        -------
        Dictionary, representing the object state to be pickled. Ignores
        the self.logger field and only returns the logger name.
        """
        return {'name': self.name}

    def __setstate__(self, state: typing.Dict[str, typing.Any]) -> None:
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


def format_array(input_array: typing.Union[np.ndarray, list]):
    """
    Transform a np array to a list of format so that it can be printed by logger
    """
    if np.size(input_array) == 1:
        return f"{input_array.item():4f}"
    format_list = []
    if isinstance(input_array, np.ndarray):
        input_array = input_array.tolist()
    for item in input_array:
        # https://stackoverflow.com/a/33482726
        format_list.append(f"{item:4f}")
    return format_list
