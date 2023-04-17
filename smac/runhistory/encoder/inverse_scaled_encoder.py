from __future__ import annotations

from typing import Any

import numpy as np

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryInverseScaledEncoder(RunHistoryEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._instances is not None and len(self._instances) > 1:
            raise NotImplementedError("Handling more than one instance is not supported for inverse scaled cost.")

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform the response values by linearly scaling
        them between zero and one and then use inverse scaling.
        """
        min_y = self._min_y - (
            self._percentile - self._min_y
        )  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        # prevent diving by zero

        min_y[np.where(min_y == self._max_y)] *= 1 - 10**-10

        values = (values - min_y) / (self._max_y - min_y)
        values = 1 - 1 / values
        return values
