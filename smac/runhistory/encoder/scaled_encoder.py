from __future__ import annotations

import numpy as np

from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryScaledEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transforms the response values by linearly scaling them between zero and one."""
        min_y, max_y = self._scaling_bounds(1 - 10**-101)
        values = (values - min_y) / (max_y - min_y)
        return values

    def _inverse_response_values(self, values: np.ndarray) -> np.ndarray:
        min_y, max_y = self._scaling_bounds(1 - 10**-101)
        return values * (max_y - min_y) + min_y
