from __future__ import annotations

import numpy as np
import sklearn.gaussian_process.kernels as kernels
from ConfigSpace import Configuration

from smac.acquisition.function.expected_improvement import EI
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.facade.abstract_facade import AbstractFacade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensifier.intensifier import Intensifier
from smac.main.config_selector import ConfigSelector
from smac.model.gaussian_process.abstract_gaussian_process import (
    AbstractGaussianProcess,
)
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.gaussian_process.kernels import (
    ConstantKernel,
    HammingKernel,
    MaternKernel,
    WhiteKernel,
)
from smac.model.gaussian_process.mcmc_gaussian_process import MCMCGaussianProcess
from smac.model.gaussian_process.priors import HorseshoePrior, LogNormalPrior
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.random_design.probability_design import ProbabilityRandomDesign
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario
from smac.utils.configspace import get_types

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class BlackBoxFacade(AbstractFacade):
    def _validate(self) -> None:
        """Ensure that the SMBO configuration with all its (updated) dependencies is valid."""
        super()._validate()
        # Activate predict incumbent
        # self.solver.epm_chooser.predict_x_best = True

        if self._scenario.instance_features is not None and len(self._scenario.instance_features) > 0:
            raise NotImplementedError("The Black-Box GP cannot handle instances.")

        if not isinstance(self._model, AbstractGaussianProcess):
            raise ValueError("The Black-Box facade only works with Gaussian Processes")

    @staticmethod
    def get_model(
        scenario: Scenario,
        *,
        model_type: str | None = None,
        kernel: kernels.Kernel | None = None,
    ) -> AbstractGaussianProcess:
        """Returns a Gaussian Process surrogate model.

        Parameters
        ----------
        scenario : Scenario
        model_type : str | None, defaults to None
            Which Gaussian Process model should be chosen. Choose between `vanilla` and `mcmc`.
        kernel : kernels.Kernel | None, defaults to None
            The kernel used in the surrogate model.

        Returns
        -------
        model : GaussianProcess | MCMCGaussianProcess
            The instantiated gaussian process.
        """
        available_model_types = [None, "vanilla", "mcmc"]
        if model_type not in available_model_types:
            types = [str(t) for t in available_model_types]
            raise ValueError(f"The model_type `{model_type}` is not supported. Choose one of {', '.join(types)}")

        if kernel is None:
            kernel = BlackBoxFacade.get_kernel(scenario=scenario)

        if model_type is None or model_type == "vanilla":
            return GaussianProcess(
                configspace=scenario.configspace,
                kernel=kernel,
                normalize_y=True,
                seed=scenario.seed,
            )
        elif model_type == "mcmc":
            n_mcmc_walkers = 3 * len(kernel.theta)
            if n_mcmc_walkers % 2 == 1:
                n_mcmc_walkers += 1

            return MCMCGaussianProcess(
                configspace=scenario.configspace,
                kernel=kernel,
                n_mcmc_walkers=n_mcmc_walkers,
                chain_length=250,
                burning_steps=250,
                normalize_y=True,
                seed=scenario.seed,
            )
        else:
            raise ValueError("Unknown model type %s" % model_type)

    @staticmethod
    def get_kernel(scenario: Scenario) -> kernels.Kernel:
        """Returns a kernel for the Gaussian Process surrogate model.

        The kernel is a composite of kernels depending on the type of hyperparameters:
        categorical (HammingKernel), continuous (Matern), and noise kernels (White).
        """
        types, _ = get_types(scenario.configspace, instance_features=None)
        cont_dims = np.where(np.array(types) == 0)[0]
        cat_dims = np.where(np.array(types) != 0)[0]

        if (len(cont_dims) + len(cat_dims)) != len(scenario.configspace.get_hyperparameters()):
            raise ValueError(
                "The inferred number of continuous and categorical hyperparameters "
                "must equal the total number of hyperparameters. Got "
                f"{(len(cont_dims) + len(cat_dims))} != {len(scenario.configspace.get_hyperparameters())}."
            )

        # Constant Kernel
        cov_amp = ConstantKernel(
            2.0,
            constant_value_bounds=(np.exp(-10), np.exp(2)),
            prior=LogNormalPrior(
                mean=0.0,
                sigma=1.0,
                seed=scenario.seed,
            ),
        )

        # Continuous / Categorical Kernels
        exp_kernel, ham_kernel = 0.0, 0.0
        if len(cont_dims) > 0:
            exp_kernel = MaternKernel(
                np.ones([len(cont_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cont_dims))],
                nu=2.5,
                operate_on=cont_dims,
            )
        if len(cat_dims) > 0:
            ham_kernel = HammingKernel(
                np.ones([len(cat_dims)]),
                [(np.exp(-6.754111155189306), np.exp(0.0858637988771976)) for _ in range(len(cat_dims))],
                operate_on=cat_dims,
            )

        # Noise Kernel
        noise_kernel = WhiteKernel(
            noise_level=1e-8,
            noise_level_bounds=(np.exp(-25), np.exp(2)),
            prior=HorseshoePrior(scale=0.1, seed=scenario.seed),
        )

        # Continuous and categecorical HPs
        if len(cont_dims) > 0 and len(cat_dims) > 0:
            kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel

        # Only continuous HPs
        elif len(cont_dims) > 0 and len(cat_dims) == 0:
            kernel = cov_amp * exp_kernel + noise_kernel

        # Only categorical HPs
        elif len(cont_dims) == 0 and len(cat_dims) > 0:
            kernel = cov_amp * ham_kernel + noise_kernel

        else:
            raise ValueError("The number of continuous and categorical hyperparameters must be greater than zero.")

        return kernel

    @staticmethod
    def get_acquisition_function(  # type: ignore
        scenario: Scenario,
        *,
        xi: float = 0.0,
    ) -> EI:
        """Returns an Expected Improvement acquisition function.

        Parameters
        ----------
        scenario : Scenario
        xi : float, defaults to 0.0
            Controls the balance between exploration and exploitation of the
            acquisition function.
        """
        return EI(xi=xi)

    @staticmethod
    def get_acquisition_maximizer(  # type: ignore
        scenario: Scenario,
        *,
        challengers: int = 1000,
        local_search_iterations: int = 10,
    ) -> LocalAndSortedRandomSearch:
        """Returns local and sorted random search as acquisition maximizer.

        Parameters
        ----------
        challengers : int, defaults to 1000
            Number of challengers.
        local_search_iterations: int, defaults to 10
            Number of local search iterations.
        """
        return LocalAndSortedRandomSearch(
            configspace=scenario.configspace,
            challengers=challengers,
            local_search_iterations=local_search_iterations,
            seed=scenario.seed,
        )

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        max_config_calls: int = 3,
        max_incumbents: int = 20,
    ) -> Intensifier:
        """Returns ``Intensifier`` as intensifier. Uses the default configuration for ``race_against``.

        Parameters
        ----------
        scenario : Scenario
        max_config_calls : int, defaults to 3
            Maximum number of configuration evaluations. Basically, how many instance-seed keys should be evaluated at
            maximum for a configuration.
        max_incumbents : int, defaults to 10
            How many incumbents to keep track of in the case of multi-objective.
        """
        return Intensifier(
            scenario=scenario,
            max_config_calls=max_config_calls,
            max_incumbents=max_incumbents,
        )

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 8,
        max_ratio: float = 0.25,
        additional_configs: list[Configuration] = None,
    ) -> SobolInitialDesign:
        """Returns a Sobol design instance.

        Parameters
        ----------
        scenario : Scenario
        n_configs : int | None, defaults to None
            Number of initial configurations (disables the arguments ``n_configs_per_hyperparameter``).
        n_configs_per_hyperparameter: int, defaults to 8
            Number of initial configurations per hyperparameter. For example, if my configuration space covers five
            hyperparameters and ``n_configs_per_hyperparameter`` is set to 10, then 50 initial configurations will be
            samples.
        max_ratio: float, defaults to 0.25
            Use at most ``scenario.n_trials`` * ``max_ratio`` number of configurations in the initial design.
            Additional configurations are not affected by this parameter.
        additional_configs: list[Configuration], defaults to []
            Adds additional configurations to the initial design.
        """
        if additional_configs is None:
            additional_configs = []
        return SobolInitialDesign(
            scenario=scenario,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_ratio=max_ratio,
            additional_configs=additional_configs,
            seed=scenario.seed,
        )

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        probability: float = 0.08447232371720552,
    ) -> ProbabilityRandomDesign:
        """Returns ``ProbabilityRandomDesign`` for interleaving configurations.

        Parameters
        ----------
        probability : float, defaults to 0.08447232371720552
            Probability that a configuration will be drawn at random.
        """
        return ProbabilityRandomDesign(seed=scenario.seed, probability=probability)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
        *,
        objective_weights: list[float] | None = None,
    ) -> MeanAggregationStrategy:
        """Returns the mean aggregation strategy for the multi-objective algorithm.

        Parameters
        ----------
        scenario : Scenario
        objective_weights : list[float] | None, defaults to None
            Weights for averaging the objectives in a weighted manner. Must be of the same length as the number of
            objectives.
        """
        return MeanAggregationStrategy(
            scenario=scenario,
            objective_weights=objective_weights,
        )

    @staticmethod
    def get_runhistory_encoder(
        scenario: Scenario,
    ) -> RunHistoryEncoder:
        """Returns the default runhistory encoder."""
        return RunHistoryEncoder(scenario)

    @staticmethod
    def get_config_selector(
        scenario: Scenario,
        *,
        retrain_after: int = 1,
        retries: int = 16,
    ) -> ConfigSelector:
        """Returns the default configuration selector."""
        return super(BlackBoxFacade, BlackBoxFacade).get_config_selector(
            scenario, retrain_after=retrain_after, retries=retries
        )
