import sys
import typing
import numpy as np

from ConfigSpace.configuration_space import Configuration

from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier

from smac.tae.execute_ta_run import ExecuteTARun
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.scenario.scenario import Scenario
from smac.tae.execute_ta_run import StatusType
from smac.runhistory.runhistory import RunHistory
from smac.utils import constants


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class MultiConfigInitialDesign(InitialDesign):

    def __init__(self,
                 tae_runner: ExecuteTARun,
                 scenario: Scenario,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 runhistory: RunHistory,
                 rng: np.random.RandomState,
                 get_configs: typing.Callable, 
                 intensifier: Intensifier,
                 aggregate_func: typing.Callable
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
        runhistory: RunHistory
            runhistory with all target algorithm runs
        rng: np.random.RandomState
            random state
        get_configs: typing.Callable
            callable to get a list of initial configurations
        intensifier: Intensifier
            intensification object to issue a racing to decide the current
            incumbent
        aggregate_func: typing:Callable
            function to aggregate performance of a configuration across instances
            
        '''
        super().__init__(tae_runner=tae_runner,
                         scenario=scenario,
                         stats=stats,
                         traj_logger=traj_logger,
                         runhistory=runhistory,
                         rng=rng)
        
        self.get_configs = get_configs
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func

    def run(self) -> Configuration:
        '''
            runs the initial design given the configurations from self.get_configs
            
            Returns
            -------
            incumbent: Configuration()
                initial incumbent configuration
        '''
        
        configs = self.get_configs()
        inc, inc_perf = self.intensifier.intensify(challengers=configs[1:], 
                                              incumbent=configs[0], 
                                              run_history=self.runhistory, 
                                              aggregate_func=self.aggregate_func)

        return inc