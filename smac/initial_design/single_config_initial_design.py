import sys

import numpy as np

from smac.initial_design.initial_design import InitialDesign
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.utils import constants

__author__ = "Marius Lindauer, Katharina Eggensperger"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class SingleConfigInitialDesign(InitialDesign):

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState
                 ):
        '''
        Constructor

        Arguments
        ---------
        tae_runner: ExecuteTARun
            target algorithm execution object
        scenario: Scenario
            scenario with all meta information (including configuration space)
        stats: Stats
            statistics of experiments; needed in case initial design already exhaust the budget
        traj_logger: TrajLogger
            trajectory logging to add new incumbents found by the initial design
        rng: np.random.RandomState
            random state
        '''
        super().__init__(tae_runner=tae_runner,
                         scenario=scenario,
                         stats=stats,
                         traj_logger=traj_logger,
                         rng=rng)

    def run(self):
        '''
            runs the initial design by calling the target algorithm
            and adding new entries to the trajectory logger

            Returns
            -------
            incumbent: Configuration()
                initial incumbent configuration
        '''

        initial_incumbent = self._select_configuration()

        # add this incumbent right away to have an entry to time point 0
        self.traj_logger.add_entry(train_perf=2**31,
                                   incumbent_id=1,
                                   incumbent=initial_incumbent)

        rand_inst = self.rng.choice(self.scenario.train_insts)

        if self.scenario.deterministic:
            initial_seed = 0
        else:
            initial_seed = self.rng.randint(0, constants.MAXINT)

        status, cost, runtime, additional_info = self.tae_runner.start(
            initial_incumbent,
            instance=rand_inst,
            cutoff=self.scenario.cutoff,
            seed=initial_seed,
            instance_specific=self.scenario.instance_specific.get(rand_inst, "0"))

        self.stats.inc_changed += 1  # first incumbent

        self.traj_logger.add_entry(train_perf=cost,
                                   incumbent_id=self.stats.inc_changed,
                                   incumbent=initial_incumbent)

        return initial_incumbent

    def _select_configuration(self):
        '''
            selects a single configuration to run

            Returns
            -------
            config: Configuration()
                initial incumbent configuration
        '''

        raise NotImplementedError
