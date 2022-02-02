import numpy as np
from typing import Optional
from smac.optimizer.multi_objective.aggregation_strategy import AggregationStrategy


class ParEGO(AggregationStrategy):
    def __init__(
        self,
        num_obj: int,
        rng: Optional[np.random.RandomState] = None,
        rho: float = 0.05,
    ):
        super(ParEGO, self).__init__(num_obj=num_obj, rng=rng)
        self.rho = rho

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

        # Then we have to compute the weight
        theta = self.rng.rand(self.num_obj)

        # Normalize st all theta values sum up to 1
        theta = theta / (np.sum(theta) + 1e-10)

        # Weight the values
        theta_f = theta * values

        return np.max(theta_f, axis=1) + self.rho * np.sum(theta_f, axis=1)
