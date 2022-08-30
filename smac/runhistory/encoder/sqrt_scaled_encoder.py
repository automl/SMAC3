from __future__ import annotations

import abc
from typing import List, Mapping, Optional, Tuple

import numpy as np

from smac import constants
from smac.configspace import convert_configurations_to_array
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.utils import normalize_costs
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.runhistory.runhistory import RunHistory, TrialKey, TrialValue
from smac.runner.runner import StatusType
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class RunHistorySqrtScaledEncoder(RunHistoryEncoder):
    """TODO."""

    def __init__(self, **kwargs):  # type: ignore[no-untyped-def]  # noqa F723
        super().__init__(**kwargs)
        if self.instances is not None and len(self.instances) > 1:
            raise NotImplementedError("Handling more than one instance is not supported for sqrt scaled cost.")

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Transform the response values by linearly scaling
        them between zero and one and then using the square root.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        # prevent diving by zero

        min_y[np.where(min_y == self.max_y)] *= 1 - 10**-10

        values = (values - min_y) / (self.max_y - min_y)
        values = np.sqrt(values)
        return values
