from __future__ import annotations

from abc import abstractmethod

import itertools

import numpy as np
import pytest
from sklearn import datasets


class Dataset:
    @abstractmethod
    def get_instances(self, n: int = 45) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_instance_features(self, n: int = 45) -> dict[str, list[int | float]]:
        raise NotImplementedError

    @abstractmethod
    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError


class DigitsDataset(Dataset):
    def __init__(self) -> None:
        self._data = datasets.load_digits()

    def get_instances(self, n: int = 45) -> list[str]:
        """Create instances from the dataset which include two classes only."""
        return [f"{classA}-{classB}" for classA, classB in itertools.combinations(self._data.target_names, 2)][:n]

    def get_instance_features(self, n: int = 45) -> dict[str, list[int | float]]:
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
        indices = np.where(np.logical_or(int(classA) == self._data.target, int(classB) == self._data.target))

        data = self._data.data[indices]
        target = self._data.target[indices]

        return data, target


@pytest.fixture
def digits_dataset() -> DigitsDataset:
    return DigitsDataset()
