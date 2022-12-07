from __future__ import annotations
from typing import Any
import numpy as np
from abc import abstractmethod


class Dataset:
    def __init__(self) -> None:
        self._data = None

    @property
    def data(self) -> Any:
        assert self._data is not None
        return self._data


class InstanceDataset:
    @abstractmethod
    def get_instances(self, n: int = 45) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    def get_instance_features(self, n: int = 45) -> dict[str, list[int | float]]:
        raise NotImplementedError

    @abstractmethod
    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError
