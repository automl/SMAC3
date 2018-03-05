import logging
import os
import shutil
import typing

import numpy as np

from smac.tae.execute_ta_run_old_hydra import ExecuteTARunOldHydra
from smac.scenario.scenario import Scenario
from smac.facade.smac_facade import SMAC

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Hydra(object):
    """Facade to use Hydra default mode

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents
    """

    def __init__(self,
                 scenario: Scenario,
                 n_iterations:int,
                 **kwargs):
        """Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        n_iterations: int,
            number of Hydra iterations
        """
        
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        
        self.n_iterations = n_iterations

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        
        self.scenario = scenario
        self.kwargs = kwargs
        

    def optimize(self):
        """Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        ----------
        portfolio : typing.List[Configuration]
            Portfolio of found configurations
        """

        portfolio = []
        smac = SMAC(scenario=self.scenario, **self.kwargs)
        portfolio_cost = np.inf
        for i in range(self.n_iterations):
            self.logger.info("Iteration: %d" %(i+1))
            
            incumbent = smac.solver.run()
            smac.stats.print_stats()
            self.logger.info("Incumbent of %d-th Iteration" %(i+1))
            self.logger.info(incumbent)
            portfolio.append(incumbent)
            
            # validate incumbent on all trainings instances
            new_rh = smac.validate(config_mode='inc', 
                                  instance_mode='train', 
                                  repetitions=1, 
                                  use_epm=False, 
                                  n_jobs=1)
            self.logger.info("Number of validated runs: %d" %(len(new_rh.data)))
            # since the TAE uses already the portfolio as an upper limit
            # the following dict already contains oracle performance
            self.logger.info("Start validation of current portfolio")
            cost_per_inst = new_rh.get_instance_costs_for_config(config=incumbent)
            cur_portfolio_cost = np.mean(list(cost_per_inst.values()))
            if portfolio_cost <= cur_portfolio_cost:
                self.logger.info("No further progress (%f) --- terminate hydra" %(portfolio_cost))
                break
            else:
                portfolio_cost = cur_portfolio_cost
                self.logger.info("Current pertfolio cost: %f" %(portfolio_cost))
            
            # modify TAE such that it return oracle performance
            #TODO: This only works for the old command line interface
            tae = ExecuteTARunOldHydra(ta=self.scenario.ta, run_obj=self.scenario.run_obj,
                                       cost_oracle=cost_per_inst)
            
            self.scenario.ta_run_limit = 20 
            
            smac = SMAC(scenario=self.scenario, tae_runner=tae, **self.kwargs)
            
        return portfolio

