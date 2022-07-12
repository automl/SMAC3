from __future__ import annotations

from dataclasses import dataclass

import ConfigSpace
import numpy as np


@dataclass
class Config:
    """
    Replacing the scenario class in the original code.


    """

    configspace: ConfigSpace

    output_directory: str | None = None
    determinstic: bool = True

    # Original: "run_obj"
    # runtime -> time
    # quality -> performance
    # Question: How to deal with categories here? -> post_init
    # But we should get rid of time
    objective: str = "performance"
    objectives: str | list[str] = "cost"
    # save_instantly: bool = False
    crash_cost: float = np.inf
    # transform_y: str | None = None  # Whether y should be transformed (different runhistory2epm)

    # Limitations
    walltime_limit: float | None = None
    cputime_limit: float | None = None
    memory_limit: float | None = None
    algorithm_walltime_limit: float | None = None
    n_runs: int = 200

    # always_race_default
    # How to deal with instances? Have them here too? It's not really a config, rather data
    # So they probably should go to the cli directory.
    # However, we need an interface to accept instances/test instances in the main python code.

    # Algorithm Configuration
    instance_features: np.array | None = None
    train_instances: np.array | None = None

    # Others
    seed: int = 0

    def __post_init__(self) -> None:
        """Checks whether the config is valid."""
        transform_y_options = [None, "log", "log_scaled", "inverse_scaled"]
        if self.transform_y not in transform_y_options:
            raise RuntimeError(f"`transform_y` must be one of `{transform_y_options}`")

    def write(self) -> None:
        pass

    @staticmethod
    def read() -> Config:
        pass
