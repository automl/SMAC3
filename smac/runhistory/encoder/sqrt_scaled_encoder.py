from __future__ import annotations

from typing import Any

import numpy as np

from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistorySqrtScaledEncoder(RunHistoryEncoder):
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        if self._instances is not None and len(self._instances) > 1:
            raise NotImplementedError("Handling more than one instance is not supported for sqrt scaled cost.")

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform the response values by linearly scaling them between zero and one and then using the
        square root.
        """
        min_y, max_y = self._scaling_bounds(1 - 10**-10)
        values = (values - min_y) / (max_y - min_y)
        values = np.sqrt(values)

        return values

    def _inverse_response_values(self, values: np.ndarray) -> np.ndarray:
        # The square root is never negative, so neither is a value it can be the root of.
        min_y, max_y = self._scaling_bounds(1 - 10**-10)
        return np.square(np.maximum(values, 0.0)) * (max_y - min_y) + min_y
