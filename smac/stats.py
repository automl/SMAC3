from __future__ import annotations

import dataclasses
import json
import time

from ConfigSpace.configuration_space import Configuration

from smac.runhistory.dataclasses import TrajectoryItem
from smac.scenario import Scenario
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class Stats:
    """Statistics which are collected during the optimization process.

    Parameters
    ----------
    scenario : Scenario
    """

    def __init__(self, scenario: Scenario):
        self._scenario = scenario
        self.reset()

    @property
    def submitted(self) -> int:
        """How many trials have been submitted."""
        return self._submitted

    @property
    def finished(self) -> int:
        """How many trials have been evaluated."""
        return self._finished

    @property
    def n_configs(self) -> int:
        """How many different configurations have been evaluated."""
        return self._n_configs

    @property
    def incumbent_changed(self) -> int:
        """How often the incumbent (aka best found configuration) changed."""
        return self._incumbent_changed

    def reset(self) -> None:
        """Resets the internal variables."""
        self._submitted = 0
        self._finished = 0
        self._n_configs = 0
        self._incumbent_changed = 0
        self._walltime_used = 0.0
        self._target_function_walltime_used = 0.0

        # Trajectory
        self._trajectory: list[TrajectoryItem] = []

        # Debug stats
        self._n_configs_per_intensify = 0
        self._n_calls_of_intensify = 0

        # Exponential moving average
        self._ema_n_configs_per_intensifiy = 0.0
        self._EMA_ALPHA = 0.2

        self._start_time = 0.0

    def add_incumbent(
        self,
        cost: float | list[float],
        incumbent: Configuration,
        budget: float | None = None,
    ) -> None:
        """Adds a new incumbent to the trajectory.

        Parameters
        ----------
        cost : float | list[float]
            Cost(s) of the incumbent.
        incumbent : Configuration
            The incumbent configuration.
        budget : float | None, defaults to None
            The used budget for the incumbent.
        """
        self._incumbent_changed += 1
        item = TrajectoryItem(
            incumbent=incumbent,
            cost=cost,
            budget=budget,
            walltime_used=self.get_used_walltime(),
            num_trial=self._finished,
        )
        self._trajectory.append(item)

    def get_incumbent(self) -> Configuration | None:
        """Returns the incumbent configuration.

        Returns
        -------
        incumbent : Configuration | None
            The incumbent configuration if it is available.
        """
        if self._incumbent_changed == 0:
            return None

        # Transform dictionary to configuration
        incumbent = self._trajectory[-1].incumbent
        return Configuration(self._scenario.configspace, values=incumbent)

    def start_timing(self) -> None:
        """Starts the timer (for the runtime configuration budget). Substracting wallclock time so we can continue
        loaded stats.
        """
        self._start_time = time.time() - self._walltime_used

    def get_used_walltime(self) -> float:
        """Returns used wallclock time."""
        return time.time() - self._start_time

    def get_remaing_walltime(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock time."""
        return self._scenario.walltime_limit - (time.time() - self._start_time)

    def get_remaining_cputime(self) -> float:
        """Subtracts the target function running budget with the used time."""
        return self._scenario.cputime_limit - self._target_function_walltime_used

    def get_remaining_trials(self) -> int:
        """Subtract the target function runs in the scenario with the used ta runs."""
        return self._scenario.n_trials - self._submitted

    def is_budget_exhausted(self) -> bool:
        """Check whether the the remaining walltime, cputime or trials was exceeded."""
        A = self.get_remaing_walltime() < 0
        B = self.get_remaining_cputime() < 0
        C = self.get_remaining_trials() < 0

        return A or B or C

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
        """Prints all statistics."""
        log = logger.info
        if debug:
            log = logger.debug

        log(
            "\n"
            f"--- STATISTICS -------------------------------------\n"
            f"--- Incumbent changed: {self._incumbent_changed - 1}\n"
            f"--- Submitted trials: {self._submitted} / {self._scenario.n_trials}\n"
            f"--- Finished trials: {self._finished} / {self._scenario.n_trials}\n"
            f"--- Configurations: {self._n_configs}\n"
            f"--- Used wallclock time: {round(self.get_used_walltime())} / {self._scenario.walltime_limit} sec\n"
            "--- Used target function runtime: "
            f"{round(self._target_function_walltime_used, 2)} / {self._scenario.cputime_limit} sec\n"
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
        """Save all relevant attributes to a json file."""
        data = {
            "submitted": self._submitted,
            "finished": self._finished,
            "n_configs": self._n_configs,
            "walltime_used": self.get_used_walltime(),
            "target_function_walltime_used": self._target_function_walltime_used,
            "incumbent_changed": self._incumbent_changed,
            "trajectory": [dataclasses.asdict(item) for item in self._trajectory],
        }

        assert self._scenario.output_directory
        filename = self._scenario.output_directory
        filename.mkdir(parents=True, exist_ok=True)
        filename = filename / "stats.json"

        logger.debug(f"Saving stats to `{filename}`")

        with open(filename, "w") as fh:
            json.dump(data, fh, indent=4)

    def load(self) -> None:
        """Load all attributes from a file into the stats object."""
        assert self._scenario.output_directory
        filename = self._scenario.output_directory / "stats.json"

        with open(filename, "r") as fh:
            data = json.load(fh)

        self._submitted = data["submitted"]
        self._finished = data["finished"]
        self._n_configs = data["n_configs"]
        self._walltime_used = data["walltime_used"]
        self._target_function_walltime_used = data["target_function_walltime_used"]
        self._incumbent_changed = data["incumbent_changed"]
        self._trajectory = [TrajectoryItem(**item) for item in data["trajectory"]]
