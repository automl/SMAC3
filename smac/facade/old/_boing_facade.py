# from __future__ import annotations

# from typing import Callable, Dict, Type

# import numpy as np
# from botorch.models.kernels.categorical import CategoricalKernel
# from gpytorch.constraints.constraints import Interval
# from gpytorch.kernels import MaternKernel, ScaleKernel
# from gpytorch.likelihoods.gaussian_likelihood import GaussianLikelihood
# from gpytorch.priors import HorseshoePrior, LogNormalPrior

# from smac.acquisition.maximizer import AbstractAcquisitionMaximizer, LocalAndSortedRandomSearch
# from smac.acquisition.function import EI, AbstractAcquisitionFunction
# from smac.main.boing import BOinGSMBO
# from smac.model.random_forest.random_forest import RandomForest
# from smac.random_design.probability_design import ProbabilityRandomDesign
# from smac.facade.hyperparameter_optimization_facade import HyperparameterOptimizationFacade
# from smac.model.abstract_model import AbstractModel
# from smac.model.gaussian_process.gpytorch_gaussian_process import (
#     GloballyAugmentedLocalGaussianProcess,
#     GPyTorchGaussianProcess,
# )
# from smac.runhistory.encoder.boing_encoder import (
#     RunHistoryRawEncoder,
#     RunHistoryRawScaledEncoder,
# )
# from smac.runner.abstract_runner import AbstractRunner
# from smac.scenario import Scenario


# class BOinGFacade(HyperparameterOptimizationFacade):
#     """
#     SMAC wrapper for BO inside Grove(BOinG):
#         Deng and Lindauer, Searching in the Forest for Local Bayesian Optimization
#         https://arxiv.org/abs/2111.05834

#     BOiNG is a two-stages optimizer: at the first stage, the global optimizer extracts the global optimum with a
#     random
#     forest (RF) model. Then in the second stage, the optimizer constructs a local model in the vicinity of the
#     configuration suggested by the global surrogate model.

#     Its Hyperparameter settings follow the implementation from smac.facade.smac_bb_facade.SMAC4BB:
#     Hyperparameters are chosen according to the best configuration for Gaussian process maximum likelihood found in
#     "Towards Assessing the Impact of Bayesian Optimization's Own Hyperparameters" by Lindauer et al., presented at the
#     DSO workshop 2019 (https://arxiv.org/abs/1908.06674).

#     Parameters
#     ----------
#     model_local: Type[AbstractModel],
#         local empirical performance model, used in subspace. Since the subspace might have different amount of
#         hyperparameters compared to the search space. We only instantiate them under the subspace.
#     model_local_kwargs: Optional[Dict] = None,
#         parameters for initializing a local model
#     acquisition_func_local: AbstractAcquisitionFunction,
#         local acquisition function,  used in subspace
#     acquisition_func_local_kwargs: Optional[Dict] = None,
#         parameters for initializing a local acquisition function optimizer
#     acq_optimizer_local: Optional[AcquisitionFunctionMaximizer] = None,
#         Optimizer of acquisition function of local models, same as above
#     acq_optimizer_local_kwargs: typing: Optional[Dict] = None,
#         parameters for the optimizer of acquisition function of local models
#     max_configs_local_fracs : float
#         The maximal number of fractions of samples to be included in the subspace. If the number of samples in the
#         subspace is greater than this value and n_min_config_inner, the subspace will be cropped to fit the
#         requirement
#     min_configs_local: int,
#         Minimum number of samples included in the inner loop model
#     do_switching: bool
#        if we want to switch between turbo and boing or do a pure BOinG search
#     turbo_kwargs: Optional[Dict] = None
#        parameters for building a turbo optimizer. For details, please refer to smac.utils.subspace.turbo
#     """

#     def __init__(
#         self,
#         scenario: Scenario,
#         target_function: AbstractRunner | Callable,
#         *,
#         local_model: AbstractModel | None = None,
#         local_acquisition_function: AbstractAcquisitionFunction | None = None,
#         local_acquisition_maximizer: AbstractAcquisitionMaximizer | None = None,
#         # model_local: Type[AbstractModel] = GloballyAugmentedLocalGaussianProcess,
#         # model_local_kwargs: Dict | None = None,
#         # acquisition_func_local: AbstractAcquisitionFunction | Type[AbstractAcquisitionFunction] = EI,
#         # acquisition_func_local_kwargs: Dict | None = None,
#         # acq_optimizer_local: AbstractAcquisitionMaximizer | None = None,
#         # acq_optimizer_local_kwargs: Dict | None = None,
#         # max_configs_local_fracs: float = 0.5,
#         # min_configs_local: int | None = None,
#         # do_switching: bool = False,
#         # turbo_kwargs: Dict | None = None,
#         **kwargs,
#     ):
#         if local_model is None:
#             local_model = self.get_local_model(scenario)

#         if local_acquisition_function is None:
#             local_acquisition_function = self.get_local_acquisition_function(scenario)

#         if local_acquisition_maximizer is None:
#             local_acquisition_maximizer = self.get_local_acquisition_maximizer(scenario)

#         self._local_model = local_model
#         self._local_acquisition_function = local_acquisition_function
#         self._local_acquisition_maximizer = local_acquisition_maximizer

#         # self.model_local = model_local
#         # if model_local_kwargs is None and issubclass(model_local, GPyTorchGaussianProcess):
#         #     model_local_kwargs = self.get_lgpga_local_components()

#         # self.model_local_kwargs = model_local_kwargs
#         # self.acquisition_func_local = acquisition_func_local
#         # self.acquisition_func_local_kwargs = acquisition_func_local_kwargs
#         # self.acq_optimizer_local = acq_optimizer_local
#         # self.acq_optimizer_local_kwargs = acq_optimizer_local_kwargs
#         # self.max_configs_local_fracs = max_configs_local_fracs
#         # self.min_configs_local = min_configs_local
#         # self.do_switching = do_switching
#         # self.turbo_kwargs = turbo_kwargs

#         super().__init__(scenario=scenario, target_function=target_function, **kwargs)

#     def _update_dependencies(self) -> None:
#         super()._update_dependencies()

#         self._local_acquisition_function.model = self._local_model
#         self._local_acquisition_maximizer.acquisition_function = self._local_acquisition_function

#     def _validate(self) -> None:
#         super()._validate()

#         if not isinstance(self._model, RandomForest):
#             raise ValueError("BOinG only supports RandomForestWithInstances as its global optimizer")

#         if not isinstance(self._runhistory_encoder, RunHistoryRawEncoder):
#             raise ValueError("BOinG only supports RunHistory2EPM4CostWithRaw as its rh transformer")

#     @staticmethod
#     def get_local_model(scenario: Scenario) -> AbstractGaussianProcess:
#         # The lower bound and upper bounds are set to be identical as SMAC4BB
#         cont_kernel_kwargs = {
#             "lengthscale_constraint": Interval(
#                 np.exp(-6.754111155189306), np.exp(0.0858637988771976), transform=None, initial_value=1.0
#             ),
#         }
#         cat_kernel_kwargs = {
#             "lengthscale_constraint": Interval(
#                 np.exp(-6.754111155189306), np.exp(0.0858637988771976), transform=None, initial_value=1.0
#             ),
#         }
#         scale_kernel_kwargs = {
#             "outputscale_constraint": Interval(np.exp(-10.0), np.exp(2.0), transform=None, initial_value=2.0),
#             "outputscale_prior": LogNormalPrior(0.0, 1.0),
#         }

#         kernel_kwargs = {
#             "cont_kernel": MaternKernel,
#             "cont_kernel_kwargs": cont_kernel_kwargs,
#             "cat_kernel": CategoricalKernel,
#             "cat_kernel_kwargs": cat_kernel_kwargs,
#             "scale_kernel": ScaleKernel,
#             "scale_kernel_kwargs": scale_kernel_kwargs,
#         }

#         # by setting lower bound of noise_constraint we could make it more stable
#         noise_prior = HorseshoePrior(0.1)
#         likelihood = GaussianLikelihood(
#             noise_prior=noise_prior, noise_constraint=Interval(1e-5, np.exp(2), transform=None)
#         ).double()

#         # ... Instantiate GPyTorchGaussianProcess

#     @staticmethod
#     def get_local_acquisition_function(  # type: ignore
#         scenario: Scenario,
#         *,
#         xi: float = 0.0,
#     ) -> EI:
#         """Returns an Expected Improvement acquisition function.

#         Parameters
#         ----------
#         scenario : Scenario
#         xi : float, defaults to 0.0
#             Controls the balance between exploration and exploitation of the
#             acquisition function.
#         """
#         return EI(xi=xi)

#     @staticmethod
#     def get_local_acquisition_maximizer(  # type: ignore
#         scenario: Scenario,
#         *,
#         challengers: int = 1000,
#         local_search_iterations: int = 10,
#     ) -> LocalAndSortedRandomSearch:
#         """Returns local and sorted random search as acquisition maximizer.

#         Parameters
#         ----------
#         challengers : int, defaults to 1000
#             Number of challengers.
#         local_search_iterations: int, defauts to 10
#             Number of local search iterations.
#         """
#         return LocalAndSortedRandomSearch(
#             configspace=scenario.configspace,
#             challengers=challengers,
#             local_search_iterations=local_search_iterations,
#             seed=scenario.seed,
#         )

#     @staticmethod
#     def get_acquisition_maximizer(
#         scenario: Scenario,
#         *,
#         local_search_iterations: int = 10,
#         challengers: int = 1000,
#     ) -> AbstractAcquisitionMaximizer:
#         """Returns the acquisition optimizer instance for finding the next candidate configuration
#         based on the acquisition function."""
#         return LocalAndSortedRandomSearch(
#             configspace=scenario.configspace,
#             local_search_iterations=local_search_iterations,
#             challengers=challengers,
#             seed=scenario.seed,
#         )

#     @staticmethod
#     def get_runhistory_encoder(scenario: Scenario) -> RunHistoryRawEncoder:
#         return RunHistoryRawScaledEncoder(scenario, scale_percentage=5)

#     @staticmethod
#     def get_random_design(scenario: Scenario, *, probability: float = 0.08447232371720552) -> ProbabilityRandomDesign:
#         return super(BOinGFacade, BOinGFacade).get_random_design(scenario=scenario, probability=probability)

#     def _get_optimizer(self) -> BOinGSMBO:
#         """Configure the BOinGSMBO optimizer, that defines the particular BO loop."""
#         return BOinGSMBO(
#             scenario=self._scenario,
#             stats=self.stats,
#             runner=self.runner,
#             initial_design=self.initial_design,
#             runhistory=self.runhistory,
#             runhistory_encoder=self.runhistory_encoder,
#             intensifier=self.intensifier,
#             model=self.model,
#             acquisition_function=self.acquisition_function,
#             acquisition_optimizer=self.acquisition_optimizer,
#             local_model=self._local_model,
#             local_acquisition_function=self._local_acquisition_function,
#             local_acquisition_maximizer=self._local_acquisition_maximizer
#             random_design=self.random_design,
#             seed=self.seed,

#             # model_local=self.model_local,
#             # model_local_kwargs=self.model_local_kwargs,
#             # acquisition_func_local=self.acquisition_func_local,
#             # acq_optimizer_local_kwargs=self.acquisition_func_local_kwargs,
#             # max_configs_local_fracs=self.max_configs_local_fracs,
#             # min_configs_local=self.min_configs_local,
#             # do_switching=self.do_switching,
#             # turbo_kwargs=self.turbo_kwargs,
#         )
