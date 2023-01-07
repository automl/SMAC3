from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractRandomDesign:
    """Abstract base of helper classes to configure interleaving of random configurations in a list of challengers.

    Parameters
    ----------
    seed : int
        The random seed initializing this design.
    """

    def __init__(self, seed: int = 0):
        self._seed = seed
        self._rng = np.random.RandomState(seed=seed)

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self._seed,
        }

    def next_iteration(self) -> None:
        """Indicates the beginning of the next SMBO iteration."""
        pass

    @abstractmethod
    def check(self, iteration: int) -> bool:
        """Check, if the next configuration should be random.

        Parameters
        ----------
        iteration : int
            Number of the i-th configuration evaluated in an SMBO iteration.

        Returns
        -------
        bool
            Whether the next configuration should be random.
        """
        pass
