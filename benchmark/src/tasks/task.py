from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from src.models.model import Model
import hashlib


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
    intensifier: str | None = None  # None == default intensifier
    n_workers: int = 1

    # Plotting options
    x_log_scale: bool = True
    y_log_scale: bool = True

    @property
    def id(self) -> str:
        """Hash of the task based on the dictionary representation."""
        data = self.__dict__.copy()
        if data["model"].dataset is not None:
            data["dataset"] = data["model"].dataset.__class__.__name__
        data["model"] = data["model"].__class__.__name__

        del data["x_log_scale"]
        del data["y_log_scale"]

        return hashlib.md5(str(data).encode("utf-8")).hexdigest()
