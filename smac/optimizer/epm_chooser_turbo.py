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
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
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
        config_space = scenario.cs
        hps = config_space.get_hyperparameters()
        for hp in hps:
            if not isinstance(hp, NumericalHyperparameter):
                raise ValueError("Current TurBO Optimizer only supports Numerical Hyperparameters")
        if len(config_space.get_conditions()) > 0 or len(config_space.get_forbiddens()) > 0:
            raise ValueError("Currently TurBO does not support conditional or forbidden hyperparameters")

        self.config_space = config_space
        n_hps = len(hps)
        self.n_dims = n_hps
        self.n_init = n_init_x_params * self.n_dims
        self.num_candidate = min(100 * n_hps, n_candidate_max)

        self.failure_tol = max(failure_tol_min, n_hps)
        self.success_tol = success_tol
        self.length = length_init
        self.length_init = length_init
        self.length_min = length_min
        self.length_max = length_max
        self._restart_turbo()

    def _restart_turbo(self):
        self.logger.debug("Current length is smaller than the minimal value, we restart a new TurBO run")
        self.success_count = 0
        self.failure_count = 0

        self.num_eval_this_round = 0
        self.last_incumbent = np.inf
        self.length = self.length_init
        np.random.seed(self.rng.randint(1, 2 ** 20))
        init_vectors = pyDOE.lhs(n=self.n_dims, samples=self.n_init)
        self.init_configs = [Configuration(self.config_space, vector=init_vector) for init_vector in init_vectors]

    def _adjust_length(self, Y):
        if Y[-1] < np.min(Y[:-1]) - 1e-3 * math.fabs(np.min(Y[:-1])):
            self.logger.debug("New suggested value is better than the incumbent, we increase success_count")
            self.success_count += 1
            self.failure_count = 0
        else:
            self.logger.debug("New suggested value is worse than the incumbent, we increase failure_count")
            self.success_count = 0
            self.failure_count += 1

        if self.success_count == self.success_tol:  # Expand trust region
            self.length = min([2.0 * self.length, self.length_max])
            self.success_count = 0
            self.logger.debug(f"Subspace length expands to {self.length}")
        elif self.failure_count == self.failure_tol:  # Shrink trust region
            self.length /= 2.0
            self.failure_count = 0
            self.logger.debug(f"Subspace length shrinks to {self.length}")

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
        if len(self.init_configs) > 0:
            self.num_eval_this_round += 1
            next_init_eval = self.init_configs.pop()
            next_init_eval.origin = "TurBO Init"
            return iter([next_init_eval])

        self.logger.debug("Search for next configuration")
        X, Y, X_configurations = self._collect_data_to_train_model()

        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )
        X = X[-self.num_eval_this_round:]
        Y = Y[-self.num_eval_this_round:]

        self.model.train(X, Y)

        self._adjust_length(Y)

        if self.length <= self.length_min:
            # restart TurBO
            self._restart_turbo()
            next_init_eval = self.init_configs.pop()
            next_init_eval.origin = "TurBO Init"
            return iter([next_init_eval])

        if incumbent_value is not None:
            best_observation = incumbent_value
            x_best_array = None  # type: typing.Optional[np.ndarray]
        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            x_best_array, best_observation = self._get_x_best(self.predict_x_best, X_configurations)

        self.acquisition_func.update(
            model=self.model,
            eta=best_observation,
            incumbent_array=x_best_array,
            num_data=len(self._get_evaluated_configs()),
            X=X_configurations,
        )

        # adjust length according to kernel length
        if isinstance(self.model, (GaussianProcess, GaussianProcessMCMC)):
            if isinstance(self.model, GaussianProcess):
                kernel_length = np.exp(self.model.hypers[1:-1])
            elif isinstance(self.model, GaussianProcessMCMC):
                kernel_length = np.exp(np.mean((np.array(self.model.hypers)[:, 1:-1]), axis=0))

            kernel_length = kernel_length / kernel_length.mean()  # This will make the next line more stable
            subspace_scale = kernel_length / np.prod(np.power(kernel_length, 1.0 / len(kernel_length)))  # We now have weights.prod() = 1

            subspace_center = x_best_array
            subspace_length = self.length * subspace_scale
        else:
            subspace_center = None
            subspace_length = None

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,  # type: ignore[attr-defined] # noqa F821
            _sorted=True,
            random_configuration_chooser=self.random_configuration_chooser,
            subspace_center=subspace_center,
            subspace_length=subspace_length,
        )
        return challengers




