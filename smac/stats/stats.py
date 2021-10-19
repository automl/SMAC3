import json
import logging
import os
import time
import typing

import numpy as np

from smac.scenario.scenario import Scenario

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Stats(object):
    """
    All statistics collected during configuration run.
    Written to output-directory to be restored

    Parameters
    ----------
    submitted_ta_runs
    finished_ta_runs
    wallclock_time_used
    ta_time_used
    inc_changed
    """

    def __init__(self, scenario: Scenario):
        """Constructor

        Parameters
        ----------
        scenario : Scenario

        output_dir : str
        """
        self.__scenario = scenario

        self.submitted_ta_runs = 0
        self.finished_ta_runs = 0
        self.n_configs = 0
        self.wallclock_time_used = 0.0
        self.ta_time_used = 0.0
        self.inc_changed = 0

        # debug stats
        self._n_configs_per_intensify = 0
        self._n_calls_of_intensify = 0
        # exponential moving average
        self._ema_n_configs_per_intensifiy = 0.0
        self._EMA_ALPHA = 0.2

        self._start_time = np.NaN
        self._logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def save(self) -> None:
        """
        Save all relevant attributes to json-dictionary.
        """
        if not self.__scenario.output_dir_for_this_run:
            self._logger.debug("No scenario.output_dir: not saving stats!")
            return
        # Set used_wallclock_time
        self.wallclock_time_used = self.get_used_wallclock_time()

        data = {}

        for v in vars(self):
            if v not in ['_Stats__scenario', '_logger', '_start_time']:
                data[v] = getattr(self, v)

        path = os.path.join(
            self.__scenario.output_dir_for_this_run, "stats.json"
        )
        self._logger.debug("Saving stats to %s", path)
        with open(path, 'w') as fh:
            json.dump(data, fh)

    def load(self, fn: typing.Optional[str] = None) -> None:
        """
        Load all attributes from dictionary in file into stats-object.

        Parameters
        ----------
        fn: string or None
            Path to file to load stats from. If no path is given, the path given
            in the current scenario is used.
        """
        if not fn:
            assert self.__scenario.output_dir_for_this_run is not None  # please mypy
            fn = os.path.join(
                self.__scenario.output_dir_for_this_run, "stats.json"
            )
        with open(fn, 'r') as fh:
            data = json.load(fh)

        # Set attributes
        for key in data:
            if hasattr(self, key):
                setattr(self, key, data[key])
            else:
                raise ValueError("Stats does not recognize {}".format(key))

    def start_timing(self) -> None:
        """
        Starts the timer (for the runtime configuration budget).
        Substracting wallclock time used so we can continue loaded Stats.
        """
        if self.__scenario:
            self._start_time = time.time() - self.wallclock_time_used
        else:
            raise ValueError("Scenario is missing")

    def get_used_wallclock_time(self) -> float:
        """Returns used wallclock time

        Returns
        -------
        wallclock_time : int
            used wallclock time in sec
        """

        return time.time() - self._start_time

    def get_remaing_time_budget(self) -> float:
        """Subtracts the runtime configuration budget with the used wallclock
        time"""
        if self.__scenario:
            return self.__scenario.wallclock_limit - (time.time() - self._start_time)
        else:
            raise ValueError("Scenario is missing")

    def get_remaining_ta_runs(self) -> int:
        """Subtract the target algorithm runs in the scenario with the used ta
        runs"""
        if self.__scenario:
            return self.__scenario.ta_run_limit - self.submitted_ta_runs  # type: ignore[attr-defined] # noqa F821
        else:
            raise ValueError("Scenario is missing")

    def get_remaining_ta_budget(self) -> float:
        """Subtracts the ta running budget with the used time"""
        if self.__scenario:
            return self.__scenario.algo_runs_timelimit - self.ta_time_used
        else:
            raise ValueError('Scenario is missing')

    def is_budget_exhausted(self) -> bool:
        """Check whether the configuration budget for time budget, ta_budget
        and submitted_ta_runs is exhausted

        Returns
        -------
        exhaustedness: boolean
            true if one of the budgets is exhausted
        """
        return (self.get_remaing_time_budget() < 0 or self.get_remaining_ta_budget() < 0
                ) or self.get_remaining_ta_runs() <= 0

    def update_average_configs_per_intensify(self, n_configs: int) -> None:
        """Updates statistics how many configurations on average per used in
        intensify

        Parameters
        ----------
        n_configs: int
            number of configurations in current intensify
        """
        self._n_calls_of_intensify += 1
        self._n_configs_per_intensify += n_configs

        if self._n_calls_of_intensify == 1:
            self._ema_n_configs_per_intensifiy = float(n_configs)
        else:
            self._ema_n_configs_per_intensifiy = (1 - self._EMA_ALPHA) * self._ema_n_configs_per_intensifiy \
                + self._EMA_ALPHA * n_configs

    def print_stats(self, debug_out: bool = False) -> None:
        """Prints all statistics

        Parameters
        ---------
        debug: bool
            use logging.debug instead of logging.info if set to true
        """
        log_func = self._logger.info
        if debug_out:
            log_func = self._logger.debug

        log_func("##########################################################")
        log_func("Statistics:")
        log_func("#Incumbent changed: %d" % (self.inc_changed - 1))  # first change is default conf
        log_func(
            "#Submitted target algorithm runs: %d / %s"
            % (self.submitted_ta_runs, str(self.__scenario.ta_run_limit))  # type: ignore[attr-defined] # noqa F821
        )
        log_func(
            "#Finished target algorithm runs: %d / %s"
            % (self.finished_ta_runs, str(self.__scenario.ta_run_limit))  # type: ignore[attr-defined] # noqa F821
        )
        log_func("#Configurations: %d" % (self.n_configs))
        log_func(
            "Used wallclock time: %.2f / %.2f sec " % (time.time() - self._start_time, self.__scenario.wallclock_limit))
        log_func(
            "Used target algorithm runtime: %.2f / %.2f sec" % (self.ta_time_used, self.__scenario.algo_runs_timelimit))
        self._logger.debug("Debug Statistics:")
        if self._n_calls_of_intensify > 0:
            self._logger.debug("Average Configurations per Intensify: %.2f" % (
                self._n_configs_per_intensify / self._n_calls_of_intensify))
            self._logger.debug("Exponential Moving Average of Configurations per Intensify: %.2f" % (
                self._ema_n_configs_per_intensifiy))

        log_func("##########################################################")
