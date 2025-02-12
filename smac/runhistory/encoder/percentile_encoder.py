from __future__ import annotations

import numpy as np
import scipy.stats
from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RunHistoryPercentileEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transforms the response values by applying a log transformation
        and ranking transformation followed by inverse Gaussian CDF scaling."""
        
        if np.any(values <= 0):
            logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use log cost." % constants.MINIMAL_COST_FOR_LOG
            )
            values[values < constants.MINIMAL_COST_FOR_LOG] = constants.MINIMAL_COST_FOR_LOG

        # Apply log transformation
        log_values = np.log(values)

        # Compute rank-based quantiles using percentileofscore
        VERY_SMALL_NUMBER = 1e-10  # Ensuring values remain in valid range
        quants = [scipy.stats.percentileofscore(log_values, v)/100 - VERY_SMALL_NUMBER for v in log_values]
        
        # Inverse Gaussian CDF transformation
        output = np.array([scipy.stats.norm.ppf(q) for q in quants]).reshape((-1, 1))
        
        return output
