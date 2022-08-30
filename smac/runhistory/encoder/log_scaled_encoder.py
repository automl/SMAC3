from __future__ import annotations

import numpy as np

import warnings
from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryLogScaledEncoder(RunHistoryEncoder):
    """TODO."""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transform the response values by linearly scaling them between zero and one and
        then using the log transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below

        # linear scaling
        # prevent diving by zero
        min_y[np.where(min_y == self.max_y)] *= 1 - 10**-10

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)

            values = (values - min_y) / (self.max_y - min_y)
            return np.log(values)
