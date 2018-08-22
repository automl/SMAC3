import logging
import os
import typing

import numpy as np

from smac.facade.smac_facade import SMAC
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    RandomConfigurationChooser, ChooserCosineAnnealing, \
    ChooserProb
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.optimizer.acquisition import EI, LogEI, AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer
from smac.tae.execute_ta_run import StatusType
from smac.epm.rf_with_instances_hpo import RandomForestWithInstancesHPO
from smac.utils.util_funcs import get_types
from smac.utils.constants import MAXINT
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.factorial_design import FactorialInitialDesign
from smac.initial_design.sobol_design import SobolDesign 

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class BORF(SMAC):
    """
    Facade to use BORF default mode
    
    see smac.facade.smac_Facade for API
    This facade overwrites option available via the SMAC facade

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

    def __init__(self,**kwargs):
        """
        Constructor
        see ~smac.facade.smac_facade for docu
        """
        
        super().__init__(**kwargs)
        self.logger.info(self.__class__)
        
        # Initial design
        # Latin Hyper Cube
        #=======================================================================
        # self.solver.initial_design = LHDesign(runhistory=self.solver.runhistory,
        #                                       intensifier=self.solver.intensifier,
        #                                       aggregate_func=self.solver.aggregate_func,
        #                                       tae_runner=self.solver.intensifier.tae_runner,
        #                                       scenario=self.solver.scenario,
        #                                       stats=self.solver.stats,
        #                                       traj_logger=self.solver.intensifier.traj_logger,
        #                                       rng=self.solver.rng)
        #=======================================================================
        # Factorial Design
        #=======================================================================
        # self.solver.initial_design = FactorialInitialDesign(runhistory=self.solver.runhistory,
        #                                                   intensifier=self.solver.intensifier,
        #                                                   aggregate_func=self.solver.aggregate_func,
        #                                                   tae_runner=self.solver.intensifier.tae_runner,
        #                                                   scenario=self.solver.scenario,
        #                                                   stats=self.solver.stats,
        #                                                   traj_logger=self.solver.intensifier.traj_logger,
        #                                                   rng=self.solver.rng)
        #=======================================================================
        # Sobol Design
        #=======================================================================
        # self.solver.initial_design = SobolDesign(runhistory=self.solver.runhistory,
        #                                           intensifier=self.solver.intensifier,
        #                                           aggregate_func=self.solver.aggregate_func,
        #                                           tae_runner=self.solver.intensifier.tae_runner,
        #                                           scenario=self.solver.scenario,
        #                                           stats=self.solver.stats,
        #                                           traj_logger=self.solver.intensifier.traj_logger,
        #                                           rng=self.solver.rng)
        #=======================================================================
        
        #== static RF settings
        self.solver.model.rf_opts.num_trees = 10
        self.solver.model.rf_opts.do_bootstrapping = True
        self.solver.model.rf_opts.tree_opts.max_features = self.solver.model.types.shape[0]
        self.solver.model.rf_opts.tree_opts.min_samples_to_split = 2
        self.solver.model.rf_opts.tree_opts.min_samples_in_leaf = 1
        
        # RF with HPO
        #=======================================================================
        # scenario = self.solver.scenario
        # types, bounds = get_types(scenario.cs, scenario.feature_array)
        # # TODO: We don't support instances here?
        # model = RandomForestWithInstancesHPO(types=types,
        #                                       bounds=bounds,
        #                                       seed=self.solver.rng.randint(MAXINT),
        #                                       log_y=scenario.logy)
        # self.solver.model = model
        #=======================================================================
        
        # no random configurations
        #rand_chooser = ChooserNoCoolDown(10**10)
        #self.solver.random_configuration_chooser = rand_chooser
        
        # random configuration with given probability
        rand_chooser = ChooserProb(prob=0.2, rng=self.solver.rng)
        self.solver.random_configuration_chooser = rand_chooser
        
        # only 1 configuration per SMBO iteration
        self.solver.scenario.intensification_percentage = 1e-10
        self.solver.intensifier.min_chall = 1
        
        # optimize in log-space
        num_params = len(self.solver.scenario.cs.get_hyperparameters())
        self.solver.rh2EPM = RunHistory2EPM4LogCost(
                    scenario=self.solver.scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, StatusType.CRASHED ],
                    impute_censored_data=False,
                    impute_state=None)
        
        # use LogEI
        #=======================================================================
        # self.solver.acquisition_func = LogEI(model=self.solver.model)
        # self.solver.acq_optimizer = InterleavedLocalAndRandomSearch(
        #         acquisition_function=self.solver.acquisition_func,
        #         config_space=self.solver.scenario.cs,
        #         rng=self.solver.rng,
        #         max_steps=self.solver.scenario.sls_max_steps,
        #         n_steps_plateau_walk=self.solver.scenario.sls_n_steps_plateau_walk
        #     )
        # self.solver.model.log_y = True
        #=======================================================================
        
        # better improve acqusition function optimization
        # 1. increase number of sls iterations
        self.solver.acq_optimizer.n_sls_iterations = 100
        # 2. more randomly sampled configurations 
        self.solver.scenario.acq_opt_challengers = 10000
        
        # activate predict incumbent
        self.solver.predict_incumbent = True
    