from __future__ import annotations

from abc import abstractmethod

from ConfigSpace import ConfigurationSpace
from src.datasets.dataset import Dataset


class Model:
    def __init__(self, dataset: Dataset | None):
        self._dataset = dataset

    @property
    def dataset(self) -> Dataset | None:
        return self._dataset

    @property
    @abstractmethod
    def configspace(self) -> ConfigurationSpace:
        raise NotImplementedError


class SingleObjectiveModel(Model):
    @abstractmethod
    def train(self) -> float:
        raise NotImplementedError


class MultiObjectiveModel(Model):
    @abstractmethod
    def train(self) -> list[float]:
        raise NotImplementedError
