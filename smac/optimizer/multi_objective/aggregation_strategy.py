import typing
import numpy as np
from abc import abstractmethod


class AggregationStrategy:
    def __init__(self, rng: typing.Optional[np.random.RandomState] = None):
        if rng is not None:
            rng = np.random.RandomState(0)
        self.rng = rng

    @abstractmethod
    def __call__(self, values: np.ndarray):
        raise NotImplementedError


class ParEGO(AggregationStrategy):
    def __init__(self,
                 rho: float = 0.05,
                 rng: typing.Optional[np.random.RandomState] = None):
        super(ParEGO).__init__(rng=rng)
        self.rho = rho

    def __call__(self, values: np.ndarray):
        """
        Transform a multi-objective loss to a single objective- loss
        Parameters
        ----------
            values: np.ndarray[num_evaluations, num_obj]
        """
        theta = self.rng.rand(self.num_obj)
        theta = theta / (np.sum(theta) + 1e-10)
        theta_f = theta * values

        return np.max(theta_f, axis=1) + np.sum(theta_f, axis=1)