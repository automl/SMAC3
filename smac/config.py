from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import ConfigSpace
import numpy as np


@dataclass(frozen=True)
class Config:
    """
    Replaces the scenario class in the original code.
    """

    configspace: ConfigSpace
    output_directory: Path | None = None
    deterministic: bool = True  # Whether the algorithm is determinstic or not

    # Original: "run_obj"
    # runtime -> time
    # quality -> performance
    # Question: How to deal with categories here? -> post_init
    # But we should get rid of time
    # objective: str = "performance"

    objectives: str | list[str] = "cost"
    # save_instantly: bool = False
    crash_cost: float = np.inf
    # transform_y: str | None = None  # Whether y should be transformed (different runhistory2epm)

    # Limitations
    walltime_limit: float = np.inf
    cputime_limit: float = np.inf
    memory_limit: float | None = None
    algorithm_walltime_limit: float | None = None
    n_runs: int = 10

    # Other time things
    intensify_percentage: float = 0.5  #

    # always_race_default
    # How to deal with instances? Have them here too? It's not really a config, rather data
    # So they probably should go to the cli directory.
    # However, we need an interface to accept instances/test instances in the main python code.

    # Algorithm Configuration
    instances: np.array | None = None
    instance_features: np.array | None = None
    instance_specifics: Mapping[str, str] | None = None

    # Others
    seed: int = 0
    n_workers: int = 1

    def __post_init__(self) -> None:
        """Checks whether the config is valid."""
        # transform_y_options = [None, "log", "log_scaled", "inverse_scaled"]
        # if self.transform_y not in transform_y_options:
        #    raise RuntimeError(f"`transform_y` must be one of `{transform_y_options}`")

        # Intensify percentage must be between 0 and 1
        assert self.intensify_percentage >= 0.0 and self.intensify_percentage <= 1.0

        if self.output_directory is None:
            # Since the dataclass is frozen, we have to hack around this.
            object.__setattr__(self, "output_directory", Path("smac3_output"))

        # Create folder
        assert self.output_directory
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def count_objectives(self) -> int:
        if isinstance(self.objectives, list):
            return len(self.objectives)

        return 1

    def write(self) -> None:
        pass

    @staticmethod
    def read() -> Config:
        pass
