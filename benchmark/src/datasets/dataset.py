from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np


class Dataset:
    def __init__(self) -> None:
        self._data = None

    @property
    def data(self) -> Any:
        assert self._data is not None
        return self._data

    def get_instances(self, n: int = None) -> list[str]:
        raise NotImplementedError

    def get_instance_features(self, n: int = None) -> dict[str, list[int | float]]:
        raise NotImplementedError

    def get_instance_data(self, instance: str) -> tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    @abstractmethod
    def get_X(self) -> np.ndarray:
        raise NotImplementedError

    @abstractmethod
    def get_Y(self) -> np.ndarray:
        raise NotImplementedError
