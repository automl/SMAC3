from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np

from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractMultiObjectiveAlgorithm(ABC):
    """A general interface for multi-objective optimizer, depending on different strategies.

    Parameters
    ----------
    scenario : Scenario
    seed : int | None, defaults to None
    """

    def __init__(
        self,
        scenario: Scenario,
        seed: int | None = None,
    ):
        if seed is None:
            seed = scenario.seed

        self._n_objectives = scenario.count_objectives()
        self._seed = seed
        self._rng = np.random.RandomState(seed)

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self._seed,
        }

    def update_on_iteration_start(self) -> None:
        """Update the internal state on start of each SMBO iteration."""
        pass

    @abstractmethod
    def __call__(self, values: list[float]) -> float:
        """Transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized values.

        Returns
        -------
        cost : float
            Combined cost.
        """
        raise NotImplementedError
