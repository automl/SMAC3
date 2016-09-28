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
                 tae_runner:ExecuteTARun,
                 scenario:Scenario,
                 stats:Stats,
                 traj_logger:TrajLogger,
                 runhistory: RunHistory,
                 rng:np.random.RandomState
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
        '''
        
        self.tae_runner = tae_runner
        self.scenario = scenario
        self.stats = stats
        self.traj_logger = traj_logger
        self.runhistory = runhistory
        self.rng = rng
        
    def run(self):
        '''
            as an initial design: it simply runs the default configuration on random pair of instance and random seed
            
            Returns
            -------
            incumbent: Configuration()
                initial incumbent configuration
        '''

        raise NotImplementedError