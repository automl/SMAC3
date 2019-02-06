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
from smac.epm.rf_with_instances import RandomForestWithInstances
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

    def __init__(self, **kwargs):
        """
        Constructor
        see ~smac.facade.smac_facade for docu
        """

        scenario = kwargs['scenario']
        
        if scenario.initial_incumbent not in ['LHD', 'FACTORIAL', 'SOBOL']:
            scenario.initial_incumbent = 'SOBOL'
        
        if scenario.transform_y is 'NONE':
            scenario.transform_y = "LOGS"

        # only 1 configuration per SMBO iteration
        intensifier_kwargs = kwargs.get('intensifier_kwargs', dict())
        intensifier_kwargs['min_chall'] = 1
        kwargs['intensifier_kwargs'] = intensifier_kwargs
        scenario.intensification_percentage = 1e-10

        model = RandomForestWithInstances
        kwargs['model'] = model

        # == static RF settings
        model_kwargs = kwargs.get('model_kwargs', dict())
        model_kwargs['num_trees'] = 10
        model_kwargs['do_bootstrapping'] = True
        model_kwargs['ratio_features'] = 1.0
        model_kwargs['min_samples_split'] = 2
        model_kwargs['min_samples_leaf'] = 1
        model_kwargs['log_y'] = True
        kwargs['model_kwargs'] = model_kwargs

        # == Acquisition function
        kwargs['acquisition_function'] = LogEI

        kwargs['runhistory2epm'] = RunHistory2EPM4LogScaledCost

        # assumes random chooser for random configs
        random_config_chooser_kwargs = kwargs.get('random_configuration_chooser_kwargs', dict())
        random_config_chooser_kwargs['prob'] = 0.0
        kwargs['random_configuration_chooser_kwargs'] = random_config_chooser_kwargs

        # better improve acquisition function optimization
        # 1. increase number of sls iterations
        acquisition_function_optimizer_kwargs = kwargs.get('acquisition_function_optimizer_kwargs', dict())
        acquisition_function_optimizer_kwargs['n_sls_iterations'] = 100
        kwargs['acquisition_function_optimizer_kwargs'] = acquisition_function_optimizer_kwargs

        super().__init__(**kwargs)
        self.logger.info(self.__class__)

        # better improve acquisition function optimization
        # 2. more randomly sampled configurations
        self.solver.scenario.acq_opt_challengers = 10000

        # activate predict incumbent
        self.solver.predict_incumbent = True
