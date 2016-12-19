import numpy as np

from smac.initial_design.single_config_initial_design import\
    SingleConfigInitialDesign
from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class RandomConfiguration(SingleConfigInitialDesign):

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
            statistics of experiments; needed in case initial design already
            exhaust the budget
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

    def _select_configuration(self):
        '''
            selects a single configuration to run

            Returns
            -------
            config: Configuration()
                initial incumbent configuration
        '''

        return self.scenario.cs.sample_configuration()
