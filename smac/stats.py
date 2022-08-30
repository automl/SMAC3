from __future__ import annotations
import dataclasses
import json
import time

import numpy as np
from ConfigSpace.configuration_space import Configuration
from smac.runhistory.dataclasses import TrajectoryItem

from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class Stats:
    """
    All statistics collected during run.
    """

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self.reset()

    @property
    def submitted(self) -> int:
        return self._submitted

    @property
    def finished(self) -> int:
        return self._finished

    @property
    def n_configs(self) -> int:
        return self._n_configs

    @property
    def incumbent_changed(self) -> int:
        return self._incumbent_changed

    def reset(self) -> None:
        self._submitted = 0
        self._finished = 0
        self._n_configs = 0
        self._incumbent_changed = 0
        self._walltime_used = 0.0
        self._target_algorithm_walltime_used = 0.0

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
        self._incumbent_changed += 1
        item = TrajectoryItem(
            cost=cost,
            incumbent=incumbent,
            walltime_used=self.get_used_walltime(),
            target_algorithm_walltime_used=self._target_algorithm_walltime_used,
            target_algorithm_runs=self._finished,
            budget=budget,
        )
        self.trajectory.append(item)

    def get_incumbent(self) -> Configuration | None:
        if self._incumbent_changed == 0:
            return None

        # Transform dictionary to configuration
        incumbent = self.trajectory[-1].incumbent
        return Configuration(self._scenario.configspace, values=incumbent)

    def start_timing(self) -> None:
        """Starts the timer (for the runtime configuration budget).

        Substracting wallclock time used so we can continue loaded Stats.
        """
        self._start_time = time.time() - self._walltime_used

    def get_used_walltime(self) -> float:
        """Returns used wallclock time."""
        return time.time() - self._start_time

    def get_remaing_walltime(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        return self._scenario.walltime_limit - (time.time() - self._start_time)

    def get_remaining_cputime(self) -> float:
        """Subtracts the ta running budget with the used time."""
        return self._scenario.cputime_limit - self._target_algorithm_walltime_used

    def get_remaining_trials(self) -> int:
        """Subtract the target algorithm runs in the scenario with the used ta runs."""
        return self._scenario.n_trials - self._submitted

    def is_budget_exhausted(self) -> bool:
        """Check whether the configuration budget for time budget, ta_budget and submitted
        is exhausted."""
        A = self.get_remaing_walltime() < 0 or self.get_remaining_cputime() < 0
        B = self.get_remaining_trials() <= 0

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
            f"--- Incumbent changed: {self._incumbent_changed - 1}\n"
            f"--- Submitted target algorithm runs: {self._submitted} / {self._scenario.n_trials}\n"
            f"--- Finished target algorithm runs: {self._finished} / {self._scenario.n_trials}\n"
            f"--- Configurations: {self._n_configs}\n"
            f"--- Used wallclock time: {round(self.get_used_walltime())} / {self._scenario.walltime_limit} sec\n"
            "--- Used target algorithm runtime: "
            f"{round(self._target_algorithm_walltime_used, 2)} / {self._scenario.cputime_limit} sec\n"
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
            "submitted": self._submitted,
            "finished": self._finished,
            "n_configs": self._n_configs,
            "walltime_used": self.get_used_walltime(),
            "target_algorithm_walltime_used": self._target_algorithm_walltime_used,
            "incumbent_changed": self._incumbent_changed,
            "trajectory": [dataclasses.asdict(item) for item in self.trajectory],
        }

        assert self._scenario.output_directory
        filename = self._scenario.output_directory
        filename.mkdir(parents=True, exist_ok=True)
        filename = filename / "stats.json"

        logger.debug(f"Saving stats to `{filename}`")

        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

    def load(self) -> None:
        """Load all attributes from dictionary in file into stats-object."""
        assert self._scenario.output_directory
        filename = self._scenario.output_directory / "stats.json"

        with open(filename, "r") as fh:
            data = json.load(fh)

        self._submitted = data["submitted"]
        self._finished = data["finished"]
        self._n_configs = data["n_configs"]
        self._walltime_used = data["walltime_used"]
        self._target_algorithm_walltime_used = data["target_algorithm_walltime_used"]
        self._incumbent_changed = data["incumbent_changed"]
        self.trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
