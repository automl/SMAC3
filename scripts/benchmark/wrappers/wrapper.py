from __future__ import annotations
from abc import abstractmethod
from benchmark.datasets.dataset import Dataset
from benchmark.models.model import Model

from benchmark.tasks.task import Task


class Wrapper:
    supported_versions: list[str] = []

    def __init__(self, task: Task) -> None:
        if len(task.objectives) > 2:
            raise RuntimeError("Not supported yet.")

        self._task = task

    @property
    def task(self) -> Task:
        return self._task

    @property
    def model(self) -> Model:
        return self._task.model

    @property
    def dataset(self) -> Dataset:
        return self._task.model.dataset

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_trajectory(self, sort_by: str = "trials") -> tuple[list[float], list[float]]:
        """List of x and y values of the incumbents over time. x depends on ``sort_by``."""
        raise NotImplementedError

    @abstractmethod
    def get_samples(self) -> tuple[list["Configuration"]]:
        raise NotImplementedError
