from __future__ import annotations
from abc import abstractmethod

from benchmark.tasks.task import Task


class Wrapper:
    def __init__(self, task: Task) -> None:
        self._task = task

    @abstractmethod
    @property
    def version(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def run(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_trajectory(self) -> list[float]:
        raise NotImplementedError

    @abstractmethod
    def plot(self) -> None:
        raise NotImplementedError

    @abstractmethod
    def used_walltime(self) -> float:
        raise NotImplementedError

    @abstractmethod
    def finished_trials(self) -> float:
        raise NotImplementedError
