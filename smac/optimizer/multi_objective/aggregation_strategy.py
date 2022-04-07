from abc import abstractmethod

import numpy as np

from smac.optimizer.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)


class AggregationStrategy(AbstractMultiObjectiveAlgorithm):
    """
    An abstract class to aggregate multi-objective losses to a single objective losses, which can then be utilized
    by the single-objective optimizer.
    """

    @abstractmethod
    def __call__(self, values: np.ndarray) -> float:
        """
        Transform a multi-objective loss to a single loss.

        Parameters
        ----------
            values: np.ndarray[num_evaluations, num_obj].

        Returns
        -------
            cost: float.
        """
        raise NotImplementedError


class MeanAggregationStrategy(AggregationStrategy):
    """
    A class to mean-aggregate multi-objective losses to a single objective losses,
    which can then be utilized by the single-objective optimizer.
    """

    def __call__(self, values: np.ndarray) -> float:
        """
        Transform a multi-objective loss to a single loss.

        Parameters
        ----------
            values (np.ndarray): Normalized values.

        Returns
        -------
            cost (float): Combined cost.
        """
        return np.mean(values, axis=1)
