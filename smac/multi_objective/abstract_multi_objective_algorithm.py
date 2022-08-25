from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

import numpy as np


class AbstractMultiObjectiveAlgorithm(ABC):
    """
    A general interface for multi-objective optimizer, depending on different strategies.
    """

    def __init__(self, seed: int = 0):
        self.seed = seed
        self.rng = np.random.RandomState(seed)

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self.seed,
        }

    @abstractmethod
    def __call__(self, values: list[float]) -> float:
        raise NotImplementedError
