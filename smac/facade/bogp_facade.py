import logging
import os
import typing

import numpy as np
import george

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
from smac.optimizer.default_priors import DefaultPrior
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC, GaussianProcess
from smac.utils.util_funcs import get_types


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class BOGP(SMAC):
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

    def __init__(self, model_type='gp_mcmc', **kwargs):
        """
        Constructor
        see ~smac.facade.smac_facade for docu
        """
        if 'model' not in kwargs or kwargs['model'] is None:
            self.rng = np.random.RandomState(np.random.randint(0, 10000))
            cov_amp = 2
            _, bounds = get_types(kwargs['scenario'].cs, instance_features=None)
            lower = np.array([b[0] for b in bounds])
            upper = np.array([b[1] for b in bounds])
            n_dims = lower.shape[0]

            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls,
                                                       ndim=n_dims)
            george.kernels.ExpKernel
            kernel = cov_amp * exp_kernel

            prior = DefaultPrior(len(kernel) + 1)

            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

            if model_type == "gp":
                model = GaussianProcess(kernel, prior=prior, rng=self.rng,
                                        normalize_output=False, normalize_input=True,
                                        lower=lower, upper=upper)
            elif model_type == "gp_mcmc":
                model = GaussianProcessMCMC(kernel, prior=prior,
                                            n_hypers=n_hypers,
                                            chain_length=200,
                                            burnin_steps=100,
                                            normalize_input=True,
                                            normalize_output=True,
                                            rng=self.rng, lower=lower, upper=upper)
            kwargs['model'] = model
        super().__init__(**kwargs)
        self.logger.info(self.__class__)
        
        self.solver.random_configuration_chooser.prob = 0.0
        
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
        self.solver.scenario.acq_opt_challengers = 1000
        
        # activate predict incumbent
        self.solver.predict_incumbent = True
    