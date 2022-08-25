from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractMultiObjectiveAlgorithm(ABC):
    """
    A general interface for multi-objective optimizer, depending on different strategies.
    """

    def __init__(self, scenario: Scenario, seed: int | None = None):
        if seed is None:
            seed = scenario.seed
        if type(scenario.objectives) == list:
            self.num_objectives = len(scenario.objectives)
        else:
            self.num_objectives = 1
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def update_on_iteration_start(self):
        """Update the internal state for each SMAC SMBO iteration."""
        pass

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self.seed,
        }

    @abstractmethod
    def __call__(self, values: list[float]) -> float:
        raise NotImplementedError
