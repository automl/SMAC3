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
        """Transforms the response values by applying a log transformation,
        rank-based quantile transformation, and inverse Gaussian CDF scaling."""

        # Debug: show what shape is coming in
        print(f">>> Encoder input values.shape = {values.shape}")

        # Safeguard: aggregate if values look like multiple per config
        if values.ndim > 1:
            logger.warning(
                f"Received values with shape {values.shape}, aggregating along axis=1."
            )
            values = np.mean(values, axis=1)

        # Replace non-positive values with minimal cost
        if np.any(values <= 0):
            logger.warning(
                "Got cost <= 0. Replacing by %f since we use log cost."
                % constants.MINIMAL_COST_FOR_LOG
            )
            values = np.clip(values, constants.MINIMAL_COST_FOR_LOG, None)

        # Apply log transformation
        log_values = np.log(values)

        # Compute rank-based quantiles
        eps = 1e-6  # keep strictly within (0,1)
        quants = [
            np.clip(scipy.stats.percentileofscore(log_values, v) / 100, eps, 1 - eps)
            for v in log_values
        ]

        # Inverse Gaussian CDF transformation
        output = scipy.stats.norm.ppf(quants).reshape((-1, 1))

        # Debug: show output shape
        print(f">>> Encoder output shape = {output.shape}")

        return output
