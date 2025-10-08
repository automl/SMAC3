from __future__ import annotations

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace, Float
from src.datasets.dataset import Dataset
from src.models.model import Model


class ACBranin(Model):
    def __init__(self, dataset: Dataset | None = None) -> None:
        super().__init__(dataset)

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        x1 = Float("x1", (-5, 10), default=0)  # This is defined for each instance
        x2 = Float("x2", (0, 15), default=7.5)

        # Add hyperparameters and conditions to our configspace
        cs.add([x2])

        return cs

    def train(self, config: Configuration, instance: float | str, seed: int) -> float:
        x1 = float(instance)
        x2 = config["x2"]
        a = 1.0
        b = 5.1 / (4.0 * np.pi**2)
        c = 5.0 / np.pi
        r = 6.0
        s = 10.0
        t = 1.0 / (8.0 * np.pi)

        cost = a * (x2 - b * x1**2 + c * x1 - r) ** 2 + s * (1 - t) * np.cos(x1) + s
        regret = cost - 0.397887

        return regret
