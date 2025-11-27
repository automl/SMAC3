from __future__ import annotations

import numpy as np
from sklearn.preprocessing import PowerTransformer

from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class RunHistoryPowerTransformEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Apply PowerTransformer (Yeo-Johnson) to response values.
        """
        if values.size == 0:
            logger.debug("Received empty array for transformation.")
            return values.reshape(-1, 1)

        values = values.reshape(-1, 1)
        transformer = PowerTransformer(method="yeo-johnson", standardize=True)
        return transformer.fit_transform(values)
