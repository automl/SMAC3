from __future__ import annotations
from dataclasses import dataclass

from benchmark.models.model import Model


@dataclass
class Task:
    """All information of the task."""

    name: str
    model: Model
    deterministic: bool
    objectives: list[str]
    walltime_limit: float
    n_trials: int
    min_budget: float | int | None
    max_budget: float | int | None
    optimization_type: str
