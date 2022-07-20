from __future__ import annotations

from typing import Any, Dict, Optional

import dataclasses
import json
import os
import time
from dataclasses import dataclass

import numpy as np
from ConfigSpace.configuration_space import Configuration, ConfigurationSpace
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    Constant,
    FloatHyperparameter,
    IntegerHyperparameter,
)

from smac.scenario import Scenario
from smac.utils.logging import get_logger

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

    def __init__(self, scenario: Scenario):
        self.scenario = scenario

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
            incumbent=incumbent,
            walltime_used=self.get_used_walltime(),
            target_algorithm_walltime_used=self.target_algorithm_walltime_used,
            target_algorithm_runs=self.finished,
            budget=budget,
        )
        self.trajectory.append(item)

    def get_incumbent(self) -> Configuration | None:
        if self.incumbent_changed == 0:
            return None

        # Transform dictionary to configuration
        incumbent = self.trajectory[-1].incumbent
        return Configuration(self.scenario.configspace, values=incumbent)

    def start_timing(self) -> None:
        """Starts the timer (for the runtime configuration budget).

        Substracting wallclock time used so we can continue loaded Stats.
        """
        self._start_time = time.time() - self.walltime_used

    def get_used_walltime(self) -> float:
        """Returns used wallclock time."""
        return time.time() - self._start_time

    def get_remaing_time_budget(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        if self.scenario:
            return self.scenario.walltime_limit - (time.time() - self._start_time)
        else:
            raise ValueError("Scenario is missing")

    def get_remaining_target_algorithm_runs(self) -> int:
        """Subtract the target algorithm runs in the scenario with the used ta runs."""
        return self.scenario.n_runs - self.submitted

    def get_remaining_target_algorithm_budget(self) -> float:
        """Subtracts the ta running budget with the used time."""
        return self.scenario.cputime_limit - self.target_algorithm_walltime_used

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

        log(
            "\n"
            f"--- STATISTICS -------------------------------------\n"
            f"--- Incumbent changed: {self.incumbent_changed - 1}\n"
            f"--- Submitted target algorithm runs: {self.submitted} / {self.scenario.n_runs}\n"
            f"--- Finished target algorithm runs: {self.finished} / {self.scenario.n_runs}\n"
            f"--- Configurations: {self.n_configs}\n"
            f"--- Used wallclock time: {round(self.get_used_walltime())} / {round(self.scenario.walltime_limit, 2)} sec\n"
            f"--- Used target algorithm runtime: {round(self.target_algorithm_walltime_used, 2)} / {round(self.scenario.cputime_limit, 2)} sec\n"
            f"----------------------------------------------------"
        )

        if self._n_calls_of_intensify > 0:
            logger.debug("Debug Statistics:")
            logger.debug(
                "Average Configurations per Intensify: %.2f"
                % (self._n_configs_per_intensify / self._n_calls_of_intensify)
            )
            logger.debug(
                "Exponential Moving Average of Configurations per Intensify: %.2f"
                % (self._ema_n_configs_per_intensifiy)
            )

    def save(self) -> None:
        """Save all relevant attributes to json-dictionary."""
        data = {
            "submitted": self.submitted,
            "finished": self.finished,
            "n_configs": self.n_configs,
            "walltime_used": self.get_used_walltime(),
            "target_algorithm_walltime_used": self.target_algorithm_walltime_used,
            "incumbent_changed": self.incumbent_changed,
            "trajectory": [dataclasses.asdict(item) for item in self.trajectory],
        }

        assert self.scenario.output_directory
        filename = self.scenario.output_directory / "stats.json"
        logger.debug(f"Saving stats to `{filename}`")

        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

    def load(self) -> None:
        """Load all attributes from dictionary in file into stats-object."""
        assert self.scenario.output_directory
        filename = self.scenario.output_directory / "stats.json"

        with open(filename, "r") as fh:
            data = json.load(fh)

        self.submitted = data["submitted"]
        self.finished = data["finished"]
        self.n_configs = data["n_configs"]
        self.walltime_used = data["walltime_used"]
        self.target_algorithm_walltime_used = data["target_algorithm_walltime_used"]
        self.incumbent_changed = data["incumbent_changed"]
        self.trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
