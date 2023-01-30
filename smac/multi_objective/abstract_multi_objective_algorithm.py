from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class AbstractMultiObjectiveAlgorithm(ABC):
    """A general interface for multi-objective optimizer, depending on different strategies."""

    def __init__(self) -> None:
        pass

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"name": self.__class__.__name__}

    def update_on_iteration_start(self) -> None:
        """Update the internal state on start of each SMBO iteration."""
        pass

    @abstractmethod
    def __call__(self, values: list[float]) -> float:
        """Transform a multi-objective loss to a single loss.

        Parameters
        ----------
        values : list[float]
            Normalized values in the range [0, 1].

        Returns
        -------
        cost : float
            Combined cost.
        """
        raise NotImplementedError
