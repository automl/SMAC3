from __future__ import annotations

import itertools

import numpy as np
from sklearn import datasets
from src.datasets.dataset import Dataset


class DigitsDataset(Dataset):
    def __init__(self) -> None:
        self._data = datasets.load_digits()

    def get_instances(self, n: int | None = None) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        instances = [f"{classA}-{classB}" for classA, classB in itertools.combinations(self.data.target_names, 2)]

        if n is not None:
            instances = instances[:n]

        return instances

    def get_instance_features(self, n: int | None = None) -> dict[str, list[int | float]]:
        """Returns the mean and variance of all instances as features."""
        features = {}
        for instance in self.get_instances(n):
            data, _ = self.get_instance_data(instance)
            features[instance] = [np.mean(data), np.var(data)]

        return features

    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        """Retrieve data from the passed instance."""
        # We split the dataset into two classes
        classA, classB = instance.split("-")
        indices = np.where(np.logical_or(int(classA) == self.data.target, int(classB) == self.data.target))

        data = self.data.data[indices]
        target = self.data.target[indices]

        return data, target

    def get_X(self) -> np.ndarray:
        """Return the data."""
        return self.data.data
    
    def get_Y(self) -> np.ndarray:
        return self.data.target