import logging

import numpy as np

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class InitialDesign(object):

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState
                 ):
        """
        Arguments
        ---------
        tae_runner: ExecuteTARun
            Target algorithm execution object.
        scenario: Scenario
            Scenario with all meta information (including configuration space).
        stats: Stats
            Statistics of experiments; needed in case initial design already 
            exhausts the budget.
        traj_logger: TrajLogger
            Trajectory logging to add new incumbents found by the initial 
            design.
        rng: np.random.RandomState
            Random state
        """

        self.tae_runner = tae_runner
        self.scenario = scenario
        self.stats = stats
        self.traj_logger = traj_logger
        self.rng = rng
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

    def run(self):
        """
        Run the initial design.

        Returns
        -------
        incumbent: Configuration
            Initial incumbent configuration.
        """

        raise NotImplementedError
