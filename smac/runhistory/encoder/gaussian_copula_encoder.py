from __future__ import annotations

import numpy as np

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger
import scipy

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryGaussianCopulaEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transforms the response values by using log."""
        min_log_cost = max(constants.MINIMAL_COST_FOR_LOG, 1e-10)
        
        if np.any(values <= 0):
            logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use"
                " log cost." % min_log_cost
            )
            values[values < min_log_cost] = min_log_cost

        n = max(len(values), 2)  # Ensure at least two values to avoid division by zero
        log_n = np.log(n) if n > 1 else 1e-10  # Avoid log(1) = 0

        quants = (scipy.stats.rankdata(values.flatten()) - 1) / (n - 1)

        cutoff = min(0.1, 1 / (4 * np.power(n, 0.25) * np.sqrt(np.pi * log_n)))

        quants = np.clip(quants, a_min=cutoff, a_max=1 - cutoff)

        rval = np.array([scipy.stats.norm.ppf(q) for q in quants]).reshape((-1, 1))

        return rval

