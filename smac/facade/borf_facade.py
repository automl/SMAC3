import logging
import os
import typing

import numpy as np

from smac.facade.smac_facade import SMAC
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    RandomConfigurationChooser, ChooserCosineAnnealing, \
    ChooserProb
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost, \
    RunHistory2EPM4LogScaledCost, RunHistory2EPM4SqrtScaledCost, \
    RunHistory2EPM4InvScaledCost, RunHistory2EPM4ScaledCost
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
        
        scenario = kwargs['scenario']
        
        if scenario.initial_incumbent not in ['LHD', 'FACTORIAL', 'SOBOL']:
            scenario.initial_incumbent = 'SOBOL'
        
        if scenario.transform_y is 'NONE':
            scenario.transform_y = "LOGS"
        #scenario.logy = True
        
        super().__init__(**kwargs)
        self.logger.info(self.__class__)
        
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
        # model = RandomForestWithInstancesHPO(types=types,
        #                                       bounds=bounds,
        #                                       seed=self.solver.rng.randint(MAXINT),
        #                                       log_y=scenario.logy)
        # self.solver.model = model
        #=======================================================================
        
        # assumes random chooser for random configs
        self.solver.random_configuration_chooser.prob = 0.0
        
        # only 1 configuration per SMBO iteration
        self.solver.scenario.intensification_percentage = 1e-10
        self.solver.intensifier.min_chall = 1
        
        # better improve acquisition function optimization
        # 1. increase number of sls iterations
        self.solver.acq_optimizer.n_sls_iterations = 100
        # 2. more randomly sampled configurations 
        self.solver.scenario.acq_opt_challengers = 10000
        
        # activate predict incumbent
        self.solver.predict_incumbent = True
    