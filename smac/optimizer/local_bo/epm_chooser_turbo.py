import logging
import typing

import numpy as np
import math

import pyDOE

from ConfigSpace.hyperparameters import NumericalHyperparameter

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gaussian_process import GaussianProcess
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.util_funcs import get_types
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
from smac.optimizer.local_bo.turbo_subspace import TurBOSubSpace
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserNoCoolDown
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats


class EPMChooserTurBO(EPMChooser):
    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[RandomConfigurationChooser] = ChooserNoCoolDown(2.0),
                 predict_x_best: bool = False,
                 min_samples_model: int = 1,
                 length_init: float = 0.8,
                 length_min: float = 0.5 ** 8,
                 length_max: float = 1.6,
                 success_tol: int = 3,
                 failure_tol_min: int = 4,
                 n_init_x_params: int = 2,
                 n_candidate_max: int = 5000,
                 ):
        """
        Interface to train the EPM and generate next configurations
        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: smac.stats.stats.Stats
            statistics object with configuration budgets
        runhistory: smac.runhistory.runhistory.RunHistory
            runhistory with all runs so far
        model: smac.epm.rf_with_instances.RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances)
        acq_optimizer: smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
        predict_x_best: bool
            Choose x_best for computing the acquisition function via the model instead of via the observations.
        min_samples_model: int
            Minimum number of samples to build a model
        length_init: float
            Initialized length after restarting
        length_min: float
            If the subspace length is smaller than length_min, TurBO will restart
        length_max: float
            Maximum length of subspace
        success_tol: int
            Number of successful suggestions (suggested points become incumbent) required for expanding subspace
        failure_tol_min: int
            Minimum number of failure suggestions (suggested points fails to become incumbent) required for shrinking
            subspace
        n_init_x_params: int
            how many configurations will be used at most in the initial design (X*D). Used for restarting the subspace
        n_candidate_max: int
            Maximal Number of points used as candidates
        """
        super(EPMChooserTurBO, self).__init__(scenario=scenario,
                                              stats=stats,
                                              runhistory=runhistory,
                                              runhistory2epm=runhistory2epm,
                                              model=model,
                                              acquisition_func=acquisition_func,
                                              acq_optimizer=acq_optimizer,
                                              restore_incumbent=restore_incumbent,
                                              rng=rng,
                                              random_configuration_chooser=random_configuration_chooser,
                                              predict_x_best=predict_x_best,
                                              min_samples_model=min_samples_model)
        types, bounds = get_types(self.scenario.cs, instance_features=None)

        self.turbo = TurBOSubSpace(config_space=scenario.cs,
                                   bounds=bounds,
                                   hps_types=types,
                                   model_local=model,
                                   acq_func_local=acquisition_func,
                                   length_init=length_init,
                                   length_min=length_min,
                                   length_max=length_max,
                                   success_tol=success_tol,
                                   failure_tol_min=failure_tol_min,
                                   n_init_x_params=n_init_x_params,
                                   n_candidate_max=n_candidate_max,)

    def choose_next(self, incumbent_value: float = None) -> typing.Iterator[Configuration]:
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.
        Parameters
        ----------
        incumbent_value: float
            Cost value of incumbent configuration (required for acquisition function);
            If not given, it will be inferred from runhistory or predicted;
            if not given and runhistory is empty, it will raise a ValueError.
        Returns
        -------
        Iterator
        """
        self.logger.debug("Search for next configuration")
        X, Y, X_configurations = self._collect_data_to_train_model()

        num_new_bservations = len(Y) - len(self.turbo.ss_y)

        new_observations = Y[-num_new_bservations:]
        if len(self.turbo.init_configs) > 0:
            self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])
            return self.turbo.generate_challengers()

        self.turbo.adjust_length(new_observations)

        self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])

        challengers = self.turbo.generate_challengers()
        return challengers