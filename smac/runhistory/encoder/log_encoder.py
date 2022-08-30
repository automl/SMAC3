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


class RunHistoryLogEncoder(RunHistoryEncoder):
    """TODO."""

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Transforms the response values by using a log
        transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        # ensure that minimal value is larger than 0
        if np.any(values <= 0):
            logger.warning(
                "Got cost of smaller/equal to 0. Replace by %f since we use"
                " log cost." % constants.MINIMAL_COST_FOR_LOG
            )
            values[values < constants.MINIMAL_COST_FOR_LOG] = constants.MINIMAL_COST_FOR_LOG
        values = np.log(values)
        return values
