from __future__ import annotations
from ConfigSpace import Configuration, ConfigurationSpace, Float
from benchmark.datasets.dataset import Dataset

from benchmark.models.model import Model


class HimmelblauModel(Model):
    def __init__(self, dataset: Dataset | None = None) -> None:
        super().__init__(dataset)

    @property
    def configspace(self) -> ConfigurationSpace:
        # Build Configuration Space which defines all parameters and their ranges
        cs = ConfigurationSpace(seed=0)

        # First we create our hyperparameters
        x = Float("x", (-5, 5))
        y = Float("y", (-5, 5))

        # Add hyperparameters and conditions to our configspace
        cs.add_hyperparameters([x, y])

        return cs

    def train(self, config: Configuration, seed: int) -> float:
        """Creates a SVM based on a configuration and evaluates it on the
        iris-dataset using cross-validation."""
        x = config["x"]
        y = config["y"]

        return (x**2 + y - 11) ** 2 + (x + y**2 - 7) ** 2
