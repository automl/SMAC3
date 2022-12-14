from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from benchmark.models.model import Model


@dataclass
class Task:
    """All information of the task."""

    name: str
    model: Model
    deterministic: bool
    objectives: list[str]
    n_trials: int
    optimization_type: str
    walltime_limit: float = np.inf
    retrain_after: int = 1
    min_budget: float | int | None = None
    max_budget: float | int | None = None
    max_config_calls: int | None = None  # For default intensifier
    n_seeds: int | None = None  # For SH/HB intensifier
    use_instances: bool = False
    incumbent_selection: str | None = None  # For SH/HB intensifier
    n_workers: int = 1
