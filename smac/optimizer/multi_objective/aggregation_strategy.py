import typing
import numpy as np
from abc import abstractmethod
from smac.optimizer.multi_objective.abstract_multi_objective_algorithm import AbstractMultiObjectiveAlgorithm


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


class ParEGO(AggregationStrategy):
    def __init__(self,
                 num_obj: int,
                 rng: typing.Optional[np.random.RandomState] = None,
                 rho: float = 0.05):
        super(ParEGO, self).__init__(num_obj=num_obj, rng=rng)
        self.rho = rho

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

        theta = self.rng.rand(self.num_obj)
        theta = theta / (np.sum(theta) + 1e-10)
        theta_f = theta * values

        return np.max(theta_f, axis=1) + np.sum(theta_f, axis=1)
