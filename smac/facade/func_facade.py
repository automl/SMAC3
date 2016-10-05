import logging

import numpy as np

from smac.facade.smac_facade import SMAC
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.configspace import ConfigurationSpace
from smac.tae.execute_func import ExecuteTAFunc
from smac.smbo.objective import average_cost
from smac.runhistory.runhistory import RunHistory

from ConfigSpace.hyperparameters import UniformFloatHyperparameter

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class FuncSMAC(SMAC):

    def __init__(self,
                 func: callable,
                 x0: list,
                 upper_bounds: list,
                 lower_bounds: list,
                 maxfun: int=np.inf,
                 maxtime: int=np.inf,
                 rng: np.random.RandomState=None):
        '''
        Special facade to use SMAC to optimize a Python function

        Parameters
        ----------
        func: callable
            function to be optimized
        x0: list
            initial guess
        upper_bounds: list
            upper bound of each input
        lower_bounds: list
            lower bound of each input
        maxfun:int
            maximal number of function evaluations
        maxtime:int
            maximal time to run (in sec)
        rng: np.random.RandomState
            Random number generator
        '''

        self.logger = logging.getLogger("FuncSMAC")
        
        aggregate_func = average_cost
        # initialize empty runhistory; needed to pass to TA
        runhistory = RunHistory(aggregate_func=aggregate_func)

        # create configuration space
        cs = ConfigurationSpace()
        for indx in range(0, len(x0)):
            parameter = UniformFloatHyperparameter(name="x%d" % (indx+1),
                                                   lower=lower_bounds[indx],
                                                   upper=upper_bounds[indx],
                                                   default=x0[indx])
            cs.add_hyperparameter(parameter)

        # create scenario
        scenario_dict = {"run_obj": "quality",  # we optimize quality
                         "cs": cs,  # configuration space
                         "deterministic": "true",
                         }
        if maxfun < np.inf:
            scenario_dict["runcount_limit"] = maxfun
        if maxtime < np.inf:
            scenario_dict["wallclock_limit"] = maxtime
        scenario = Scenario(scenario_dict)

        # create statistics
        self.stats = Stats(scenario)
        # register target algorithm
        taf = ExecuteTAFunc(ta=func, stats=self.stats, run_obj="quality", runhistory=runhistory )

        # use SMAC facade
        super().__init__(scenario=scenario,
                         tae_runner=taf,
                         stats=self.stats,
                         runhistory=runhistory,
                         rng=rng)
