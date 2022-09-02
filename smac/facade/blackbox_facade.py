from __future__ import annotations

import numpy as np
import sklearn.gaussian_process.kernels as kernels

from smac.acquisition import AbstractAcquisitionOptimizer
from smac.acquisition.functions import AbstractAcquisitionFunction
from smac.acquisition.functions.expected_improvement import EI
from smac.acquisition.local_and_random_search import LocalAndSortedRandomSearch
from smac.random_design.probability_design import ProbabilityRandomDesign
from ConfigSpace import Configuration
from smac.facade.facade import Facade
from smac.initial_design.sobol_design import SobolInitialDesign
from smac.intensification.intensification import Intensifier
from smac.model.gaussian_process.gaussian_process import GaussianProcess
from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
from smac.model.gaussian_process.kernels import (
    ConstantKernel,
    HammingKernel,
    Matern,
    WhiteKernel,
)
from smac.model.gaussian_process.mcmc_gaussian_process import MCMCGaussianProcess
from smac.model.gaussian_process.priors import HorseshoePrior, LogNormalPrior
from smac.model.utils import get_types
from smac.multi_objective import AbstractMultiObjectiveAlgorithm
from smac.multi_objective.aggregation_strategy import MeanAggregationStrategy
from smac.runhistory.encoder.encoder import RunHistoryEncoder
from smac.scenario import Scenario

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


class BlackBoxFacade(Facade):
    def _validate(self) -> None:
        """Ensure that the SMBO configuration with all its (updated) dependencies is valid."""
        super()._validate()
        # TODO what about these? vvv
        # self.solver.scenario.acq_opt_challengers = 1000  # type: ignore[attr-defined] # noqa F821
        # # activate predict incumbent
        # self.solver.epm_chooser.predict_x_best = True

        if self._scenario.instance_features is not None and len(self._scenario.instance_features) > 0:
            raise NotImplementedError("The Black-Box GP cannot handle instances.")

        if not isinstance(self._model, AbstractGaussianProcess):
            raise ValueError(
                "The Black-Box facade only works with Gaussian Process-"
                "like surrogate models (inheriting from smac.model.gaussian_process.BaseModel, "
                f"got type {type(self._model)}."
            )

    @staticmethod
    def get_model(
        scenario: Scenario,
        *,
        model_type: str = "gp",
        kernel: kernels.Kernel | None = None,
    ) -> AbstractGaussianProcess:
        """Returns a Gaussian Process surrogate model. Please check its documentation for
        detials."""
        available_model_types = ["gp", "gp_mcmc"]
        if model_type not in available_model_types:
            raise ValueError(f"model_type {model_type} not in available model types")

        if kernel is None:
            kernel = BlackBoxFacade.get_kernel(scenario=scenario)

        if model_type == "gp":
            return GaussianProcess(
                configspace=scenario.configspace,
                kernel=kernel,
                normalize_y=True,
                seed=scenario.seed,
            )
        elif model_type == "gp_mcmc":
            n_mcmc_walkers = 3 * len(kernel.theta)
            if n_mcmc_walkers % 2 == 1:
                n_mcmc_walkers += 1

            return MCMCGaussianProcess(
                configspace=scenario.configspace,
                kernel=kernel,
                n_mcmc_walkers=n_mcmc_walkers,
                chain_length=250,
                burnin_steps=250,
                normalize_y=True,
                seed=scenario.seed,
            )
        else:
            raise ValueError("Unknown model type %s" % model_type)

    @staticmethod
    def get_kernel(scenario: Scenario) -> kernels.Kernel:
        """Returns a kernel for the Gaussian Process surrogate model.
        The kernel is a composite of kernels depending on the type of hyperparameters:
        categorical (HammingKernel), continuous (Matern), and noise kernels (White)."""
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
                mean=0.0, sigma=1.0, seed=scenario.seed
            ),  # TODO convert expected arg RandomState -> Generator
        )

        # Continuous / Categorical Kernels
        exp_kernel, ham_kernel = 0.0, 0.0
        if len(cont_dims) > 0:
            exp_kernel = Matern(
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
        xi: float = 0.0,
    ) -> EI:
        """Returns the acquisition function instance (EI: Expected Improvement) for the Black-Box
        facade.
        """
        return EI(xi=xi)

    @staticmethod
    def get_acquisition_optimizer(  # type: ignore
        scenario: Scenario,
        *,
        local_search_iterations: int = 10,
        challengers: int = 1000,
    ) -> LocalAndSortedRandomSearch:
        """Returns the acquisition optimizer instance (LocalAndSOrtedRandomSearch) for the
        Black-Box facade. Please check its documentation for detials."""
        optimizer = LocalAndSortedRandomSearch(
            configspace=scenario.configspace,
            local_search_iterations=local_search_iterations,
            challengers=challengers,
            seed=scenario.seed,
        )
        return optimizer

    @staticmethod
    def get_intensifier(  # type: ignore
        scenario: Scenario,
        *,
        min_challenger: int = 1,
        min_config_calls: int = 1,
        max_config_calls: int = 3,
        intensify_percentage: float = 0.5,
    ) -> Intensifier:
        """Returns the Intensifier instance  for the Black-Box facade. Please check its
        documentation for details."""
        intensifier = Intensifier(
            scenario=scenario,
            min_challenger=min_challenger,
            race_against=scenario.configspace.get_default_configuration(),
            min_config_calls=min_config_calls,
            max_config_calls=max_config_calls,
            intensify_percentage=intensify_percentage,
        )

        return intensifier

    @staticmethod
    def get_initial_design(  # type: ignore
        scenario: Scenario,
        *,
        configs: list[Configuration] | None = None,
        n_configs: int | None = None,
        n_configs_per_hyperparamter: int = 10,
        max_config_ratio: float = 0.25,  # Use at most X*budget in the initial design
    ) -> SobolInitialDesign:
        """Returns a Sobol design instance for the Black-Box facade.
        Please check its documentation."""
        initial_design = SobolInitialDesign(
            scenario=scenario,
            configs=configs,
            n_configs=n_configs,
            n_configs_per_hyperparameter=n_configs_per_hyperparamter,
            max_config_ratio=max_config_ratio,
        )
        return initial_design

    @staticmethod
    def get_random_design(  # type: ignore
        scenario: Scenario,
        *,
        random_probability: float = 0.08447232371720552,
    ) -> ProbabilityRandomDesign:
        """Returns a random design instance (ProbabilityRandomDesign) for the Black-Box facade,
        detailing the interleaving of configurations derived from BO's acquisition with those
        from random search. Please check its documentation."""
        return ProbabilityRandomDesign(seed=scenario.seed, probability=random_probability)

    @staticmethod
    def get_multi_objective_algorithm(  # type: ignore
        scenario: Scenario,
    ) -> MeanAggregationStrategy:
        """Returns the multi-objective algorithm instance (MeanAggregationStrategy) for the
        Black-Box facade."""
        return MeanAggregationStrategy(scenario=scenario)

    @staticmethod
    def get_runhistory_encoder(
        scenario: Scenario,
    ) -> RunHistoryEncoder:
        return RunHistoryEncoder(scenario)
