from __future__ import annotations

from typing import Dict, Type, Union

import numpy as np
from botorch.models.kernels.categorical import CategoricalKernel
from gpytorch.constraints.constraints import Interval
from gpytorch.kernels import MaternKernel, ScaleKernel
from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
from gpytorch.priors import HorseshoePrior, LogNormalPrior

from smac.acquisition import AbstractAcquisitionOptimizer, LocalAndSortedRandomSearch
from smac.acquisition.functions import EI, AbstractAcquisitionFunction
from smac.chooser.boing_chooser import BOinGConfigurationChooser
from smac.chooser.random_chooser import ProbabilityConfigurationChooser
from smac.facade.hyperparameter_facade import HyperparameterFacade
from smac.model.base_model import BaseModel
from smac.model.gaussian_process.augmented_local_gaussian_process import (
    GloballyAugmentedLocalGaussianProcess,
)
from smac.runhistory.encoder.boing_encoder import (
    RunHistory2EPM4CostWithRaw,
    RunHistory2EPM4ScaledLogCostWithRaw,
)
from smac.scenario import Scenario


class BOinGFacade(HyperparameterFacade):
    """
    SMAC wrapper for BO inside Grove(BOinG):
        Deng and Lindauer, Searching in the Forest for Local Bayesian Optimization
        https://arxiv.org/abs/2111.05834

    BOiNG is a two-stages optimizer: at the first stage, the global optimizer extracts the global optimum with a random
    forest (RF) model. Then in the second stage, the optimizer constructs a local model in the vicinity of the
    configuration suggested by the global surrogate model.

    Its Hyperparameter settings follow the implementation from smac.facade.smac_bb_facade.SMAC4BB:
    Hyperparameters are chosen according to the best configuration for Gaussian process maximum likelihood found in
    "Towards Assessing the Impact of Bayesian Optimization's Own Hyperparameters" by Lindauer et al., presented at the
    DSO workshop 2019 (https://arxiv.org/abs/1908.06674).
    """

    @staticmethod
    def get_runhistory_encoder(scenario: Scenario) -> RunHistory2EPM4CostWithRaw:
        transformer = RunHistory2EPM4ScaledLogCostWithRaw(
            scenario=scenario,
            n_params=len(scenario.configspace.get_hyperparameters()),
            scale_percentage=5,
            seed=scenario.seed,
        )

        return transformer

    @staticmethod
    def get_random_configuration_chooser(
        scenario: Scenario,
        *,
        probability: float = 0.08447232371720552,
    ) -> ProbabilityConfigurationChooser:
        return ProbabilityConfigurationChooser(prob=probability)

    @staticmethod
    def get_acquisition_optimizer(
        scenario: Scenario,
        *,
        local_search_iterations: int = 10,
        challengers: int = 1000,
    ) -> AbstractAcquisitionOptimizer:
        optimizer = LocalAndSortedRandomSearch(
            configspace=scenario.configspace,
            local_search_iterations=local_search_iterations,
            challengers=challengers,
            seed=scenario.seed,
        )
        return optimizer

    @staticmethod
    def get_configuration_chooser(
        predict_x_best: bool = True,
        min_samples_model: int = 1,
        model_local: Type[BaseModel] = GloballyAugmentedLocalGaussianProcess,
        model_local_kwargs: Dict | None = None,
        acquisition_func_local: Union[AbstractAcquisitionFunction, Type[AbstractAcquisitionFunction]] = EI,
        acquisition_func_local_kwargs: Dict | None = None,
        acq_optimizer_local: AbstractAcquisitionOptimizer | None = None,
        acq_optimizer_local_kwargs: Dict | None = None,
        max_configs_local_fracs: float = 0.5,
        min_configs_local: int | None = None,
        do_switching: bool = False,
        turbo_kwargs: Dict | None = None,
    ):
        """
        Parameters
        ----------
        model_local: Type[BaseModel],
            local empirical performance model, used in subspace. Since the subspace might have different amount of
            hyperparameters compared to the search space. We only instantiate them under the subspace.
        model_local_kwargs: Optional[Dict] = None,
            parameters for initializing a local model
        acquisition_func_local: AbstractAcquisitionFunction,
            local acquisition function,  used in subspace
        acquisition_func_local_kwargs: Optional[Dict] = None,
            parameters for initializing a local acquisition function optimizer
        acq_optimizer_local: Optional[AcquisitionFunctionMaximizer] = None,
            Optimizer of acquisition function of local models, same as above, since an acquisition function optimizer
            requries
        acq_optimizer_local_kwargs: typing: Optional[Dict] = None,
            parameters for the optimizer of acquisition function of local models
        max_configs_local_fracs : float
            The maximal number of fractions of samples to be included in the subspace. If the number of samples in the
            subspace is greater than this value and n_min_config_inner, the subspace will be cropped to fit the requirement
        min_configs_local: int,
            Minimum number of samples included in the inner loop model
        do_switching: bool
           if we want to switch between turbo and boing or do a pure BOinG search
        turbo_kwargs: Optional[Dict] = None
           parameters for building a turbo optimizer. For details, please refer to smac.utils.subspace.turbo
        """
        if model_local_kwargs is None and model_local.__name__ == "GloballyAugmentedLocalGaussianProcess":
            model_local_kwargs = BOinGFacade.get_lgpga_local_components()

        return BOinGConfigurationChooser(
            predict_x_best=predict_x_best,
            min_samples_model=min_samples_model,
            model_local=model_local,
            model_local_kwargs=model_local_kwargs,
            acquisition_func_local=acquisition_func_local,
            acq_optimizer_local_kwargs=acquisition_func_local_kwargs,
            max_configs_local_fracs=max_configs_local_fracs,
            min_configs_local=min_configs_local,
            do_switching=do_switching,
            turbo_kwargs=turbo_kwargs,
        )

    @staticmethod
    def get_lgpga_local_components() -> Dict:
        """
        A function to construct the required components that could be implemented to construct a LGPGA model.
        """
        # The lower bound and upper bounds are set to be identical as SMAC4BB
        cont_kernel_kwargs = {
            "lengthscale_constraint": Interval(
                np.exp(-6.754111155189306), np.exp(0.0858637988771976), transform=None, initial_value=1.0
            ),
        }
        cat_kernel_kwargs = {
            "lengthscale_constraint": Interval(
                np.exp(-6.754111155189306), np.exp(0.0858637988771976), transform=None, initial_value=1.0
            ),
        }
        scale_kernel_kwargs = {
            "outputscale_constraint": Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0),
            "outputscale_prior": LogNormalPrior(0.0, 1.0),
        }

        kernel_kwargs = {
            "cont_kernel": MaternKernel,
            "cont_kernel_kwargs": cont_kernel_kwargs,
            "cat_kernel": CategoricalKernel,
            "cat_kernel_kwargs": cat_kernel_kwargs,
            "scale_kernel": ScaleKernel,
            "scale_kernel_kwargs": scale_kernel_kwargs,
        }

        # by setting lower bound of noise_constraint we could make it more stable
        noise_prior = HorseshoePrior(0.1)
        likelihood = GaussianLikelihood(
            noise_prior=noise_prior, noise_constraint=Interval(1e-5, np.exp(2), transform=None)
        ).double()
        return {
            "model_local": GloballyAugmentedLocalGaussianProcess,
            "model_local_kwargs": dict(kernel_kwargs=kernel_kwargs, likelihood=likelihood),
        }
