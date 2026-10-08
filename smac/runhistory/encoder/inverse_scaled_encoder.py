from __future__ import annotations

from typing import Any

import numpy as np

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
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
        min_y, max_y = self._scaling_bounds(1 - 10**-10)
        values = (values - min_y) / (max_y - min_y)
        values = 1 - 1 / values
        return values

    def _inverse_response_values(self, values: np.ndarray) -> np.ndarray:
        # 1 - 1/s is increasing in s and below one; a value at or above one has no cost, so it is held just below.
        values = np.minimum(values, 1 - constants.VERY_SMALL_NUMBER)
        min_y, max_y = self._scaling_bounds(1 - 10**-10)
        return 1 / (1 - values) * (max_y - min_y) + min_y
