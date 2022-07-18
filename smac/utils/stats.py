from __future__ import annotations

from dataclasses import dataclass
import dataclasses
from typing import Any, Dict, Optional

import json
import os
import time

import numpy as np
from smac.config import Config
from smac.utils.logging import get_logger
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
)

__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


@dataclass
class TrajectoryItem:
    """Replaces `TrajEntry` from the original code."""

    cost: float
    # incumbent_id: int  # TODO: Do we actually need the incumbent_id?
    incumbent: Configuration | dict[str, Any]
    walltime_used: float
    target_algorithm_walltime_used: float
    target_algorithm_runs: int
    budget: float

    def __post_init__(self) -> None:
        # Transform configuration to dict
        if isinstance(self.incumbent, Configuration):
            self.incumbent = self.incumbent.get_dictionary()


class Stats:
    """
    All statistics collected during run.
    """

    def __init__(self, config: Config):
        self.config = config

        self.submitted = 0
        self.finished = 0
        self.n_configs = 0
        self.walltime_used = 0.0
        self.target_algorithm_walltime_used = 0.0
        self.incumbent_changed = 0

        # Trajectory
        self.trajectory: list[TrajectoryItem] = []

        # Debug stats
        self._n_configs_per_intensify = 0
        self._n_calls_of_intensify = 0

        # Exponential moving average
        self._ema_n_configs_per_intensifiy = 0.0
        self._EMA_ALPHA = 0.2

        self._start_time = np.NaN

    def add_incumbent(self, cost: float, incumbent: Configuration, budget: float = 0) -> None:
        self.incumbent_changed += 1
        item = TrajectoryItem(
            cost=cost,
            # incumbent_id=self.incumbent_changed,
            incumbent=incumbent,
            walltime_used=self.get_used_wallclock_time(),
            target_algorithm_walltime_used=self.target_algorithm_walltime_used,
            target_algorithm_runs=self.finished,
            budget=budget,
        )
        self.trajectory.append(item)

    def start_timing(self) -> None:
        """Starts the timer (for the runtime configuration budget).

        Substracting wallclock time used so we can continue loaded Stats.
        """
        self._start_time = time.time() - self.walltime_used

    def get_used_wallclock_time(self) -> float:
        """Returns used wallclock time."""
        return time.time() - self._start_time

    def get_remaing_time_budget(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        if self.config:
            return self.config.walltime_limit - (time.time() - self._start_time)
        else:
            raise ValueError("Scenario is missing")

    def get_remaining_target_algorithm_runs(self) -> int:
        """Subtract the target algorithm runs in the scenario with the used ta runs."""
        return self.config.n_runs - self.submitted

    def get_remaining_target_algorithm_budget(self) -> float:
        """Subtracts the ta running budget with the used time."""
        return self.config.cputime_limit - self.target_algorithm_walltime_used

    def is_budget_exhausted(self) -> bool:
        """Check whether the configuration budget for time budget, ta_budget and submitted
        is exhausted."""
        A = self.get_remaing_time_budget() < 0 or self.get_remaining_target_algorithm_budget() < 0
        B = self.get_remaining_target_algorithm_runs() <= 0

        return A or B

    def update_average_configs_per_intensify(self, n_configs: int) -> None:
        """Updates statistics how many configurations on average per used in intensify.

        Parameters
        ----------
        n_configs : int
            Number of configurations in current intensify.
        """
        self._n_calls_of_intensify += 1
        self._n_configs_per_intensify += n_configs

        if self._n_calls_of_intensify == 1:
            self._ema_n_configs_per_intensifiy = float(n_configs)
        else:
            self._ema_n_configs_per_intensifiy = (
                1 - self._EMA_ALPHA
            ) * self._ema_n_configs_per_intensifiy + self._EMA_ALPHA * n_configs

    def print(self, debug: bool = True) -> None:
        """Prints all statistics.

        Parameters
        ----------
        debug: bool
            use logging.debug instead of logging.info if set to true
        """
        log = logger.info
        if debug:
            log = logger.debug

        log("---------------------STATISTICS---------------------")
        log(f"Incumbent changed: {self.incumbent_changed - 1}")
        log(f"Submitted target algorithm runs: {self.submitted} / {self.config.n_runs}")
        log(f"Finished target algorithm runs: {self.finished} / {self.config.n_runs}")
        log(f"Configurations: {self.n_configs}")
        log("Used wallclock time: %.2f / %.2f sec " % (time.time() - self._start_time, self.config.walltime_limit))
        log(
            "Used target algorithm runtime: %.2f / %.2f sec"
            % (self.target_algorithm_walltime_used, self.config.cputime_limit)
        )

        logger.debug("Debug Statistics:")
        if self._n_calls_of_intensify > 0:
            logger.debug(
                "Average Configurations per Intensify: %.2f"
                % (self._n_configs_per_intensify / self._n_calls_of_intensify)
            )
            logger.debug(
                "Exponential Moving Average of Configurations per Intensify: %.2f"
                % (self._ema_n_configs_per_intensifiy)
            )
        log("----------------------------------------------------")

    def save(self) -> None:
        """Save all relevant attributes to json-dictionary."""
        data = {
            "submitted": self.submitted,
            "finished": self.finished,
            "n_configs": self.n_configs,
            "walltime_used": self.walltime_used,
            "target_algorithm_walltime_used": self.target_algorithm_walltime_used,
            "incumbent_changed": self.incumbent_changed,
            "trajectory": [dataclasses.asdict(item) for item in self.trajectory],
        }

        assert self.config.output_directory
        filename = self.config.output_directory / "stats.json"
        logger.debug(f"Saving stats to `{filename}`")

        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

    def load(self) -> None:
        """Load all attributes from dictionary in file into stats-object."""
        assert self.config.output_directory
        filename = self.config.output_directory / "stats.json"

        with open(filename, "r") as fh:
            data = json.load(fh)

        self.submitted = data["submitted"]
        self.finished = data["finished"]
        self.n_configs = data["n_configs"]
        self.walltime_used = data["walltime_used"]
        self.target_algorithm_walltime_used = data["target_algorithm_walltime_used"]
        self.incumbent_changed = data["incumbent_changed"]
        self.trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
