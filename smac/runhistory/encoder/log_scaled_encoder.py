from __future__ import annotations

import warnings

import numpy as np

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RunHistoryLogScaledEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform the response values by linearly scaling them between zero and one and
        then using the log transformation.
        """
        min_y = self._compute_min_y()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            values = (values - min_y) / (self._max_y - min_y)
            return np.log(values)

    def transform_response_values_inverse(self, values: np.ndarray) -> np.ndarray:
        """Inverse transform the response values by using the exponential function and then undoing the linear scale
        between zero and one.
        """
        min_y = self._compute_min_y()

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            values = np.exp(values)
            values = values * (self._max_y - min_y) + min_y
            return values

    def _compute_min_y(self) -> np.ndarray:
        min_y = self._min_y - (self._percentile - self._min_y)
        min_y -= constants.VERY_SMALL_NUMBER
        # Linear scaling
        # prevent diving by zero
        min_y[np.where(min_y == self._max_y)] *= 1 - 10**-10
        return min_y
