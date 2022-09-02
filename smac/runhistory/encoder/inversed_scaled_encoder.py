from __future__ import annotations


import numpy as np
from typing import Any

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryInverseScaledEncoder(RunHistoryEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self.instances is not None and len(self.instances) > 1:
            raise NotImplementedError("Handling more than one instance is not supported for inverse scaled cost.")

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform the response values by linearly scaling
        them between zero and one and then using inverse scaling."""
        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        # prevent diving by zero

        min_y[np.where(min_y == self.max_y)] *= 1 - 10**-10

        values = (values - min_y) / (self.max_y - min_y)
        values = 1 - 1 / values
        return values
