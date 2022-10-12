# from __future__ import annotations

# from typing import Dict, List, Tuple, Type

# import inspect

# import numpy as np
# from ConfigSpace import ConfigurationSpace

# from smac.acquisition import AbstractAcquisitionMaximizer, LocalAndSortedRandomSearch
# from smac.acquisition.function import EI, AbstractAcquisitionFunction
# from ConfigSpace import Configuration
# from smac.model.abstract_model import AbstractModel
# from smac.model.gaussian_process.gpytorch_gaussian_process import GloballyAugmentedLocalGaussianProcess
# from smac.utils.logging import get_logger
# from smac.utils.subspaces import LocalSubspace

# logger = get_logger(__name__)


# class BOinGSubspace(LocalSubspace):
#     """
#     Subspace for BOinG optimizer. Each time we create a new epm model for the subspace and optimize to maximize the
#     acquisition function inside this subregion.

#     Parameters
#     ----------
#     acq_optimizer_local: Optional[AcquisitionFunctionMaximizer]
#         Subspace optimizer, used to give a set of suggested points. Unlike the optimizer implemented in epm_chooser,
#         this optimizer does not require runhistory objects.
#     acq_optimizer_local_kwargs
#         Parameters for acq_optimizer_local
#     """

#     def __init__(
#         self,
#         config_space: ConfigurationSpace,
#         bounds: List[Tuple[float, float]],
#         hps_types: List[int],
#         bounds_ss_cont: np.ndarray | None = None,
#         bounds_ss_cat: List[Tuple] | None = None,
#         model_local: AbstractModel | Type[AbstractModel] = GloballyAugmentedLocalGaussianProcess,
#         model_local_kwargs: Dict = {},
#         acq_func_local: AbstractAcquisitionFunction | Type[AbstractAcquisitionFunction] = EI,
#         acq_func_local_kwargs: Dict | None = None,
#         rng: np.random.RandomState | None = None,
#         initial_data: Tuple[np.ndarray, np.ndarray] | None = None,
#         activate_dims: np.ndarray | None = None,
#         incumbent_array: np.ndarray | None = None,
#         acq_optimizer_local: AbstractAcquisitionMaximizer | None = None,
#         acq_optimizer_local_kwargs: Dict | None = None,
#     ):
#         super(BOinGSubspace, self).__init__(
#             config_space=config_space,
#             bounds=bounds,
#             hps_types=hps_types,
#             bounds_ss_cont=bounds_ss_cont,
#             bounds_ss_cat=bounds_ss_cat,
#             model_local=model_local,
#             model_local_kwargs=model_local_kwargs,
#             acq_func_local=acq_func_local,
#             acq_func_local_kwargs=acq_func_local_kwargs,
#             rng=rng,
#             initial_data=initial_data,
#             activate_dims=activate_dims,
#             incumbent_array=incumbent_array,
#         )
#         if bounds_ss_cont is None and bounds_ss_cat is None:
#             self.config_origin = None  # type: ignore
#         else:
#             self.config_origin = "BOinG"
#         if isinstance(self.model, GloballyAugmentedLocalGaussianProcess):
#             num_inducing_points = min(max(min(2 * len(self.activate_dims_cont), 10), self.model_x.shape[0] // 20), 50)
#             self.model.update_attribute(num_inducing_points=num_inducing_points)

#         subspace_acq_func_opt_kwargs = {
#             "acquisition_function": self.acquisition_function,
#             "configspace": self.cs_local,  # type: ignore[attr-defined] # noqa F821
#             "seed": self.rng.randint(1, 2**20),
#         }

#         if isinstance(acq_optimizer_local, AbstractAcquisitionMaximizer):
#             # we copy the attribute of the local acquisition function optimizer but replace it with our local model
#             # setting. This helps a better exploration in the beginning.
#             for key in inspect.signature(acq_optimizer_local.__init__).parameters.keys():  # type: ignore[misc]
#                 if key == "self":
#                     continue
#                 elif key in subspace_acq_func_opt_kwargs:
#                     continue
#                 elif hasattr(acq_func_local, key):
#                     subspace_acq_func_opt_kwargs[key] = getattr(acq_func_local, key)
#             self.acq_optimizer_local = type(acq_optimizer_local)(**subspace_acq_func_opt_kwargs)
#         else:
#             if acq_optimizer_local is None:
#                 acq_optimizer_local = LocalAndSortedRandomSearch  # type: ignore
#                 if acq_optimizer_local_kwargs is not None:
#                     subspace_acq_func_opt_kwargs.update(acq_optimizer_local_kwargs)
#                 else:
#                     # Here are the setting used by squirrel-optimizer
#                     # https://github.com/automl/Squirrel-Optimizer-BBO-NeurIPS20-automlorg/blob/main/squirrel-optim
# izer/smac_optim.py
#                     local_search_iterations = {
#                         1: 10,
#                         2: 10,
#                         3: 10,
#                         4: 10,
#                         5: 10,
#                         6: 10,
#                         7: 8,
#                         8: 6,
#                     }.get(len(self.cs_local.get_hyperparameters()), 5)

#                     subspace_acq_func_opt_kwargs.update(
#                         {"n_steps_plateau_walk": 5, "local_search_iterations": local_search_iterations}
#                     )

#             elif inspect.isclass(acq_optimizer_local, AbstractAcquisitionMaximizer):
#                 subspace_acq_func_opt_kwargs.update(acq_optimizer_local_kwargs)
#             else:
#                 raise TypeError(
#                     f"subspace_optimizer must be None or an object implementing the "
#                     f"AcquisitionFunctionMaximizer, but is '{acq_optimizer_local}'"
#                 )

#             self.acq_optimizer_local = acq_optimizer_local(**subspace_acq_func_opt_kwargs)  # type: ignore

#     def _generate_challengers(self, **optimizer_kwargs: Dict) -> List[Tuple[float, Configuration]]:
#         """
#         Generate new challengers list for this subspace. This optimizer is similar to
#         smac.optimizer.ei_optimization.LocalAndSortedRandomSearch except that we don't read the past evaluated
#         information from the runhistory but directly assign new values to the
#         """
#         self.model.train(self.model_x, self.model_y)
#         self.update_model(predict_x_best=True, update_incumbent_array=True)
#         num_points_rs = 1000

#         if isinstance(self.acq_optimizer_local, LocalAndSortedRandomSearch):
#             next_configs_random = self.acq_optimizer_local.random_search._maximize(
#                 previous_configs=[],
#                 num_points=num_points_rs,
#                 _sorted=True,
#             )
#             if len(self.ss_x) == 0:
#                 init_points_local = self.cs_local.sample_configuration(size=self.acq_optimizer_local.n_sls_iterations)
#             else:
#                 previous_configs = [Configuration(configuration_space=self.cs_local, vector=ss_x) for ss_x in
# self.ss_x]
#                 init_points_local = self.acq_optimizer_local.local_search._get_init_points_from_previous_configs(
#                     self.acq_optimizer_local.local_search_iterations, previous_configs, next_configs_random
#                 )

#             configs_acq_local = self.acq_optimizer_local.local_search._do_search(init_points_local)

#             # shuffle for random tie-break
#             self.rng.shuffle(configs_acq_local)

#             # sort according to acq value
#             configs_acq_local.sort(reverse=True, key=lambda x: x[0])

#             for _, inc in configs_acq_local:
#                 inc.origin = "Local Search"

#             # Having the configurations from random search, sorted by their
#             # acquisition function value is important for the first few iterations
#             # of SMAC. As long as the random forest predicts constant value, we
#             # want to use only random configurations. Having them at the begging of
#             # the list ensures this (even after adding the configurations by local
#             # search, and then sorting them)
#             next_configs_by_acq_value = next_configs_random + configs_acq_local

#             next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
#             logger.debug(
#                 "First 5 acq func (origin) values of selected configurations: %s",
#                 str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:5]]),
#             )
#             return next_configs_by_acq_value
#         else:
#             return self.acq_optimizer_local._maximize([], num_points_rs)  # type: ignore
