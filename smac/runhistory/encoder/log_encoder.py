from __future__ import annotations

import numpy as np

from smac import constants
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistoryLogEncoder(RunHistoryEncoder):
    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transforms the response values by using log."""
        # ensure that minimal value is larger than 0
        if np.any(values <= 0):
            logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use"
                " log cost." % constants.MINIMAL_COST_FOR_LOG
            )
            values[values < constants.MINIMAL_COST_FOR_LOG] = constants.MINIMAL_COST_FOR_LOG

        return np.log(values)
