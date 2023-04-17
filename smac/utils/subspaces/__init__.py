# from __future__ import annotations

# from abc import ABC, abstractmethod
# from typing import Any, Dict, Iterator, List, Tuple, Type

# import copy
# import inspect
# import logging
# import math

# import numpy as np
# from ConfigSpace.forbidden import (
#     AbstractForbiddenComponent,
#     ForbiddenAndConjunction,
#     MultipleValueForbiddenClause,
# )
# from ConfigSpace.hyperparameters import (
#     CategoricalHyperparameter,
#     Constant,
#     Hyperparameter,
#     NumericalHyperparameter,
#     OrdinalHyperparameter,
#     UniformFloatHyperparameter,
#     UniformIntegerHyperparameter,
# )

# from smac.acquisition.function import EI, AbstractAcquisitionFunction
# from ConfigSpace import Configuration, ConfigurationSpace
# from smac.model.abstract_model import AbstractModel
# from smac.model.gaussian_process.gpytorch_gaussian_process import GloballyAugmentedLocalGaussianProcess
# from smac.model.gaussian_process.kernels._boing import construct_gp_kernel
# from smac.model.utils import check_subspace_points
# from smac.utils.logging import get_logger

# logger = get_logger(__name__)


# class LocalSubspace(ABC):
#     """
#     A subspace that is designed for local Bayesian Optimization. If bounds_ss_cont and bounds_ss_cat are not given,
#     this subspace is equivalent to the original configuration space. Additionally, this subspace
#     supports local BO that only works with a subset of the dimensions, where the missing values are filled by the
#     corresponding values from incumbent_array.

#     Parameters
#     ----------
#     config_space: ConfigurationSpace
#         raw Configuration space
#     bounds: List[Tuple[float, float]]
#         raw bounds of the Configuration space, notice that here bounds denotes the bounds of the entire space
#     hps_types: List[int],
#         types of the hyperparameters
#     bounds_ss_cont: np.ndarray(D_cont, 2)
#         subspaces bounds of continuous hyperparameters, its length is the number of continuous hyperparameters
#     bounds_ss_cat: List[Tuple]
#         subspaces bounds of categorical hyperparameters, its length is the number of categorical hyperparameters
#     rng: np.random.RandomState
#         random state
#     model_local: ~smac.epm.base_epm.BaseEPM
#         model in subspace
#     model_local_kwargs: Dict | None
#         argument for subspace model
#     acq_func_local: ~smac.optimizer.ei_optimization.AbstractAcquisitionFunction
#         local acquisition function
#     acq_func_local_kwargs: Dict | None
#         argument for acquisition function
#     activate_dims: np.ndarray | None
#         activate dimensions in the subspace, if it is None, we preserve all the dimensions
#     incumbent_array: np.ndarray | None
#         incumbent array, used when activate_dims has less dimension and this value is used to complementary the
#         resulted configurations
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
#     ):
#         self.cs_global = config_space
#         if rng is None:
#             self.rng = np.random.RandomState(1)
#         else:
#             self.rng = np.random.RandomState(rng.randint(0, 2**20))

#         n_hypers = len(config_space.get_hyperparameters())
#         model_types = copy.deepcopy(hps_types)
#         model_bounds = copy.deepcopy(bounds)

#         cat_dims = np.where(np.array(hps_types) != 0)[0]
#         cont_dims = np.where(np.array(hps_types) == 0)[0]

#         if activate_dims is None:
#             activate_dims = np.arange(n_hypers)
#             activate_dims_cont = cont_dims
#             activate_dims_cat = cat_dims
#             self.activate_dims = activate_dims
#             activate_dims_cont_ss = np.arange(len(activate_dims_cont))
#             activate_dims_cat_ss = np.arange(len(activate_dims_cat))
#         else:
#             activate_dims_cont, _, activate_dims_cont_ss = np.intersect1d(
#                 activate_dims, cont_dims, assume_unique=True, return_indices=True
#             )
#             activate_dims_cat, _, activate_dims_cat_ss = np.intersect1d(
#                 activate_dims, cat_dims, assume_unique=True, return_indices=True
#             )
#             self.activate_dims = activate_dims

#         self.activate_dims_cont = activate_dims_cont_ss
#         self.activate_dims_cat = activate_dims_cat_ss

#         lbs = np.full(n_hypers, 0.0)
#         scales = np.full(n_hypers, 1.0)

#         if bounds_ss_cont is None and bounds_ss_cat is None:
#             # cs_inner is cs
#             self.cs_local = config_space
#             self.new_config_space = False
#             self.bounds_ss_cont = np.tile([0.0, 1.0], [len(self.activate_dims_cont), 1])
#             self.bounds_ss_cat = []  # type: Optional[List[Tuple]]
#             self.lbs = lbs
#             self.scales = scales
#             self.new_config = False

#         else:
#             self.new_config = True
#             # we normalize the non-CategoricalHyperparameter by x = (x-lb)*scale

#             hps = config_space.get_hyperparameters()

#             # deal with categorical hyperaprameters
#             for i, cat_idx in enumerate(activate_dims_cat):
#                 hp_cat = hps[cat_idx]  # type: CategoricalHyperparameter
#                 parents = config_space.get_parents_of(hp_cat.name)
#                 if len(parents) == 0:
#                     can_be_inactive = False
#                 else:
#                     can_be_inactive = True
#                 if bounds_ss_cat is None:
#                     n_cats = len(hp_cat.choices)
#                 else:
#                     n_cats = len(bounds_ss_cat[i])
#                 if can_be_inactive:
#                     n_cats = n_cats + 1
#                 model_types[cat_idx] = n_cats
#                 model_bounds[cat_idx] = (int(n_cats), np.nan)

#             # store the dimensions of numerical hyperparameters, UniformFloatHyperparameter and
#             # UniformIntegerHyperparameter
#             dims_cont_num = []
#             idx_cont_num = []
#             dims_cont_ord = []
#             idx_cont_ord = []
#             ord_hps = {}

#             # deal with ordinary hyperaprameters
#             for i, cont_idx in enumerate(activate_dims_cont):
#                 param = hps[cont_idx]
#                 if isinstance(param, OrdinalHyperparameter):
#                     parents = config_space.get_parents_of(param.name)
#                     if len(parents) == 0:
#                         can_be_inactive = False
#                     else:
#                         can_be_inactive = True
#                     if bounds_ss_cont is None:
#                         n_seqs = len(param.sequence)
#                     else:
#                         n_seqs = bounds_ss_cont[i][1] - bounds_ss_cont[i][0] + 1
#                     if can_be_inactive:
#                         model_bounds[cont_idx] = (0, int(n_seqs))
#                     else:
#                         model_bounds[cont_idx] = (0, int(n_seqs) - 1)
#                     if bounds_ss_cont is None:
#                         lbs[cont_idx] = 0  # in subspace, it should start from 0
#                         ord_hps[param.name] = (0, int(n_seqs))
#                     else:
#                         lbs[cont_idx] = bounds_ss_cont[i][0]  # in subspace, it should start from 0
#                         ord_hps[param.name] = bounds_ss_cont[i]
#                     dims_cont_ord.append(cont_idx)
#                     idx_cont_ord.append(i)
#                 else:
#                     dims_cont_num.append(cont_idx)
#                     idx_cont_num.append(i)

#             if bounds_ss_cat is not None:
#                 self.bounds_ss_cat = [bounds_ss_cat[act_dims_cat_ss] for act_dims_cat_ss in activate_dims_cat_ss]
#             else:
#                 self.bounds_ss_cat = None
#             self.bounds_ss_cont = bounds_ss_cont[activate_dims_cont_ss] if bounds_ss_cont is not None else None

#             if bounds_ss_cont is None:
#                 lbs[dims_cont_num] = 0.0
#                 scales[dims_cont_num] = 1.0
#             else:
#                 lbs[dims_cont_num] = bounds_ss_cont[idx_cont_num, 0]
#                 # rescale numerical hyperparameters to [0., 1.]
#                 scales[dims_cont_num] = 1.0 / (bounds_ss_cont[idx_cont_num, 1] - bounds_ss_cont[idx_cont_num, 0])

#             self.lbs = lbs[activate_dims]
#             self.scales = scales[activate_dims]

#             self.cs_local = ConfigurationSpace()
#             hp_list = []
#             idx_cont = 0
#             idx_cat = 0

#             hps = config_space.get_hyperparameters()

#             for idx in self.activate_dims:
#                 param = hps[idx]
#                 if isinstance(param, CategoricalHyperparameter):
#                     if bounds_ss_cat is None:
#                         hp_new = copy.deepcopy(param)
#                         idx_cat += 1
#                     else:
#                         choices = [param.choices[int(choice_idx)] for choice_idx in bounds_ss_cat[idx_cat]]
#                         # cat_freq_arr = np.array((cats_freq[idx_cat]))
#                         # weights = cat_freq_arr / np.sum(cat_freq_arr)
#                         hp_new = CategoricalHyperparameter(param.name, choices=choices)  # , weights=weights)
#                         idx_cat += 1

#                 elif isinstance(param, OrdinalHyperparameter):
#                     param_seq = ord_hps.get(param.name)
#                     raw_seq = param.sequence
#                     ord_indices = np.arange(*param_seq)
#                     new_seq = [raw_seq[int(round(idx))] for idx in ord_indices]
#                     hp_new = OrdinalHyperparameter(param.name, sequence=new_seq)
#                     idx_cont += 1

#                 elif isinstance(param, Constant):
#                     hp_new = copy.deepcopy(param)
#                 elif isinstance(param, (UniformFloatHyperparameter, UniformIntegerHyperparameter)):
#                     if bounds_ss_cont is None:
#                         hp_new = copy.deepcopy(param)
#                         idx_cont += 1
#                     else:
#                         if isinstance(param, UniformFloatHyperparameter):
#                             lower = param.lower
#                             upper = param.upper
#                             if param.log:
#                                 lower_log = np.log(lower)
#                                 upper_log = np.log(upper)
#                                 hp_new_lower = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] +
# lower_log)
#                                 hp_new_upper = np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] +
# lower_log)
#                                 hp_new = UniformFloatHyperparameter(
#                                     name=param.name,
#                                     lower=max(hp_new_lower, lower),
#                                     upper=min(hp_new_upper, upper),
#                                     log=True,
#                                 )
#                             else:
#                                 hp_new_lower = (upper - lower) * bounds_ss_cont[idx_cont][0] + lower
#                                 hp_new_upper = (upper - lower) * bounds_ss_cont[idx_cont][1] + lower
#                                 hp_new = UniformFloatHyperparameter(
#                                     name=param.name,
#                                     lower=max(hp_new_lower, lower),
#                                     upper=min(hp_new_upper, upper),
#                                     log=False,
#                                 )
#                             idx_cont += 1
#                         elif isinstance(param, UniformIntegerHyperparameter):
#                             lower = param.lower
#                             upper = param.upper
#                             if param.log:
#                                 lower_log = np.log(lower)
#                                 upper_log = np.log(upper)
#                                 hp_new_lower = int(
#                                     math.floor(
#                                         np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][0] + lower_log)
#                                     )
#                                 )
#                                 hp_new_upper = int(
#                                     math.ceil(np.exp((upper_log - lower_log) * bounds_ss_cont[idx_cont][1] +
# lower_log))
#                                 )

#                                 hp_new_lower_log = np.log(hp_new_lower)
#                                 hp_new_upper_log = np.log(hp_new_upper)
#                                 new_scale = (upper_log - lower_log) / (hp_new_upper_log - hp_new_lower_log)
#                                 new_lb = (hp_new_lower_log - lower_log) / (hp_new_upper_log - hp_new_lower_log)

#                                 self.scales[idx] = new_scale
#                                 self.lbs[idx] = new_lb

#                                 hp_new = UniformIntegerHyperparameter(
#                                     name=param.name,
#                                     lower=max(hp_new_lower, lower),
#                                     upper=min(hp_new_upper, upper),
#                                     log=True,
#                                 )
#                             else:
#                                 hp_new_lower = int(math.floor((upper - lower) * bounds_ss_cont[idx_cont][0])) + lower
#                                 hp_new_upper = int(math.ceil((upper - lower) * bounds_ss_cont[idx_cont][1])) + lower

#                                 new_scale = (upper - lower) / (hp_new_upper - hp_new_lower)
#                                 new_lb = (hp_new_lower - lower) / (hp_new_upper - hp_new_lower)
#                                 self.scales[idx] = new_scale
#                                 self.lbs[idx] = new_lb

#                                 hp_new = UniformIntegerHyperparameter(
#                                     name=param.name,
#                                     lower=max(hp_new_lower, lower),
#                                     upper=min(hp_new_upper, upper),
#                                     log=False,
#                                 )

#                             idx_cont += 1
#                 else:
#                     raise ValueError(f"Unsupported type of Hyperparameter: {type(param)}")
#                 hp_list.append(hp_new)

#             # We only consider plain hyperparameters
#             self.cs_local.add_hyperparameters(hp_list)
#             forbiddens_ss = []
#             forbiddens = config_space.get_forbiddens()
#             for forbidden in forbiddens:
#                 forbiden_ss = self.fit_forbidden_to_ss(cs_local=self.cs_local, forbidden=forbidden)
#                 if forbiden_ss is not None:
#                     forbiddens_ss.append(forbiden_ss)
#             if len(forbiddens_ss) > 0:
#                 self.cs_local.add_forbidden_clauses(forbiddens_ss)

#         model_kwargs = dict(
#             configspace=self.cs_local,
#             # types=[model_types[activate_dim] for activate_dim in activate_dims] if model_types is not None else
# None,
#             # bounds=[model_bounds[activate_dim] for activate_dim in activate_dims] if model_bounds is not None
#             # else None,
#             bounds_cont=np.array([[0, 1.0] for _ in range(len(activate_dims_cont))]),
#             bounds_cat=self.bounds_ss_cat,
#             seed=self.rng.randint(0, 2**20),
#         )

#         if inspect.isclass(model_local):
#             model_local_kwargs_copy = copy.deepcopy(model_local_kwargs)
#             if "kernel_kwargs" in model_local_kwargs_copy:
#                 kernel_kwargs = model_local_kwargs_copy["kernel_kwargs"]
#                 kernel = construct_gp_kernel(kernel_kwargs, activate_dims_cont_ss, activate_dims_cat_ss)
#                 del model_local_kwargs_copy["kernel_kwargs"]
#                 model_local_kwargs_copy["kernel"] = kernel

#             if model_local_kwargs is not None:
#                 model_kwargs.update(model_local_kwargs_copy)

#             all_arguments = inspect.signature(model_local).parameters.keys()
#             if "bounds_cont" not in all_arguments:
#                 del model_kwargs["bounds_cont"]
#             if "bounds_cat" not in all_arguments:
#                 del model_kwargs["bounds_cat"]
#             model = model_local(**model_kwargs)  # type: ignore
#         else:
#             model = model_local

#         self.model = model

#         if inspect.isclass(acq_func_local):
#             acq_func_kwargs = {}
#             if acq_func_local_kwargs is not None:
#                 acq_func_kwargs.update(acq_func_local_kwargs)
#             acquisition_function = acq_func_local(**acq_func_kwargs)  # type: ignore
#         else:
#             acquisition_function = acq_func_local

#         self.acquisition_function = acquisition_function

#         self.incumbent_array = incumbent_array

#         self.model_x = np.empty([0, len(activate_dims)])
#         self.ss_x = np.empty([0, len(activate_dims)])
#         self.model_y = np.empty([0, 1])
#         self.ss_y = np.empty([0, 1])

#         if initial_data is not None:
#             X = initial_data[0]
#             y = initial_data[1]

#             self.add_new_observations(X, y)

#         self.config_origin = "subspace"

#     @staticmethod
#     def fit_forbidden_to_ss(
#         cs_local: ConfigurationSpace, forbidden: AbstractForbiddenComponent
#     ) -> AbstractForbiddenComponent | None:
#         """
#         Fit the forbidden to subspaces. If the target forbidden can be added to subspace, we return a new forbidden
#         with exactly the same type of the input forbidden. Otherwise, None is returned.

#         Parameters
#         ----------
#         cs_local: ConfigurationSpace
#             local configuration space of the subspace
#         forbidden: AbstractForbiddenComponent
#             forbidden to check
#         Returns
#         -------
#         forbidden_ss: AbstractForbiddenComponent | None
#             forbidden in subspaces

#         """
#         if isinstance(forbidden, ForbiddenAndConjunction):
#             forbidden_ss_components = []
#             for forbid in forbidden.components:
#                 # If any of the AndConjunction is not supported by the subspace, we simply ignore them
#                 forbid_ss = LocalSubspace.fit_forbidden_to_ss(cs_local, forbid)
#                 if forbid_ss is None:
#                     return None
#                 forbidden_ss_components.append(forbid_ss)
#             return type(forbidden)(*forbidden_ss_components)
#         else:
#             forbidden_hp_name = forbidden.hyperparameter.name
#             if forbidden_hp_name not in cs_local:
#                 return None
#             hp_ss = cs_local.get_hyperparameter(forbidden_hp_name)

#             def is_value_in_hp(value: Any, hp: Hyperparameter) -> bool:
#                 """Check if the value is in the range of the hp."""
#                 if isinstance(hp, NumericalHyperparameter):
#                     return hp.lower <= value <= hp.upper
#                 elif isinstance(hp, OrdinalHyperparameter):
#                     return value in hp.sequence
#                 elif isinstance(hp, CategoricalHyperparameter):
#                     return value in hp.choices
#                 else:
#                     raise NotImplementedError("Unsupported type of hyperparameter!")

#             if isinstance(forbidden, MultipleValueForbiddenClause):
#                 forbidden_values = forbidden.values
#                 for forbidden_value in forbidden_values:
#                     if not is_value_in_hp(forbidden_value, hp_ss):
#                         return None
#                 return type(forbidden)(hp_ss, forbidden_values)
#             else:
#                 forbidden_value = forbidden.value
#                 if is_value_in_hp(forbidden_value, hp_ss):
#                     return type(forbidden)(hp_ss, forbidden_value)
#             return None

#     def update_model(self, predict_x_best: bool = True, update_incumbent_array: bool = False) -> None:
#         """
#         Update the model and acquisition function parameters

#         Parameters
#         ----------
#         predict_x_best: bool,
#             if the incumbent is acquired by the predicted mean of a surrogate model
#         update_incumbent_array: bool
#             if the incumbent_array of this subspace is replaced with the newly updated incumbent
#         """
#         acq_func_kwargs = {"model": self.model, "num_data": len(self.ss_x)}

#         if predict_x_best:
#             try:
#                 mu, _ = self.model.predict(self.ss_x)
#             except Exception as e:
#                 # Some times it could occur that LGPGA fails to predict the mean value of ss_x because of
#                 # numerical issues
#                 logger.warning(f"Fail to predict ss_x due to {e}")
#                 mu = self.ss_y
#             idx_eta = np.argmin(mu)
#             incumbent_array = self.ss_x[idx_eta]
#             acq_func_kwargs.update({"incumbent_array": incumbent_array, "eta": mu[idx_eta]})
#         else:
#             idx_eta = np.argmin(self.ss_y)
#             incumbent_array = self.ss_x[idx_eta]
#             acq_func_kwargs.update({"incumbent_array": incumbent_array, "eta": self.ss_y[idx_eta]})
#         if update_incumbent_array:
#             if self.incumbent_array is None:
#                 self.incumbent_array = self.ss_x[idx_eta]
#             else:
#                 self.incumbent_array[self.activate_dims] = self.ss_x[idx_eta]

#         self.acquisition_function.update(**acq_func_kwargs)

#     def add_new_observations(self, X: np.ndarray, y: np.ndarray) -> None:
#         """
#         Add new observations to the subspace

#         Parameters
#         ----------
#         X: np.ndarray(N,D),
#             new feature vector of the observations, constructed by the global configuration space
#         y: np.ndarray(N)
#            new performances of the observations
#         Return
#         ----------
#         indices_in_ss:np.ndarray(N)
#             indices of data that included in subspaces
#         """
#         if len(X.shape) == 1:
#             X = X[np.newaxis, :]
#         if len(y.shape) == 1:
#             y = y[:, np.newaxis]

#         X = X[:, self.activate_dims]

#         ss_indices = check_subspace_points(
#             X=X,
#             cont_dims=self.activate_dims_cont,
#             cat_dims=self.activate_dims_cat,
#             bounds_cont=self.bounds_ss_cont,
#             bounds_cat=self.bounds_ss_cat,
#         )

#         X = self.normalize_input(X=X)

#         self.model_x = np.vstack([self.model_x, X])
#         self.model_y = np.vstack([self.model_y, y])

#         self.ss_x = np.vstack([self.ss_x, X[ss_indices]])
#         self.ss_y = np.vstack([self.ss_y, y[ss_indices]])

#     def update_incumbent_array(self, new_incumbent: np.ndarray) -> None:
#         """
#         Update a new incumbent array. The array is generated from the global configuration

#         Parameters
#         ----------
#         new_incumbent: np.ndarray(D)
#             new incumbent, which correspondences to the global configuration
#         """
#         self.incumbent_array = self.normalize_input(X=new_incumbent)

#     def generate_challengers(self, **optimizer_kwargs: Any) -> Iterator:
#         """
#         Generate a list of challengers that will be transformed into the global configuration space

#         Parameters
#         ----------
#         optimizer_kwargs: Any
#             additional configurations passed to 'self._generate_challengers'

#         Returns
#         -------
#             A list of challengers in the global configuration space

#         """
#         challengers = self._generate_challengers(**optimizer_kwargs)
#         return ChallengerListLocal(
#             cs_local=self.cs_local,
#             cs_global=self.cs_global,
#             challengers=challengers,
#             config_origin=self.config_origin,
#             incumbent_array=self.incumbent_array,
#         )

#     @abstractmethod
#     def _generate_challengers(self, **optimizer_kwargs: Any) -> List[Tuple[float, Configuration]]:
#         """Generate new challengers list for this subspace"""
#         raise NotImplementedError

#     def normalize_input(self, X: np.ndarray) -> np.ndarray:
#         """
#         Normalize X to fit the local configuration space

#         Parameters
#         ----------
#         X: np.ndarray(N,D)
#             input X, configurations arrays
#         Returns
#         -------
#         X_normalized: np.ndarray(N,D)
#             normalized input X
#         """
#         if not self.new_config:
#             return X

#         if len(X.shape) == 1:
#             X = X[np.newaxis, :]

#         # normalize X
#         X_normalized = (X - self.lbs) * self.scales
#         if self.bounds_ss_cat is not None:
#             # normalize categorical function, for instance, if bounds_subspace[i] is a categorical bound contains
#             # elements [1, 3, 5], then we map 1->0, 3->1, 5->2
#             for cat_idx, cat_bound in zip(self.activate_dims_cat, self.bounds_ss_cat):
#                 X_i = X_normalized[:, cat_idx]
#                 cond_list = [X_i == cat for cat in cat_bound]
#                 choice_list = np.arange(len(cat_bound))
#                 X_i = np.select(cond_list, choice_list)
#                 X_normalized[:, cat_idx] = X_i

#         return X_normalized


# class ChallengerListLocal(Iterator):
#     def __init__(
#         self,
#         cs_local: ConfigurationSpace,
#         cs_global: ConfigurationSpace,
#         challengers: List[Tuple[float, Configuration]],
#         config_origin: str,
#         incumbent_array: np.ndarray | None = None,
#     ):
#         """
#         A Challenger list to convert the configuration from the local configuration space to the global configuration
#          space

#         Parameters
#         ----------
#         cs_local: ConfigurationSpace
#             local configuration space
#         cs_global: ConfigurationSpace
#             global configuration space
#         challengers: List[Tuple[float, Configuration]],
#             challenger lists
#         config_origin: str
#             configuration origin
#         incumbent_array: np.ndarray | None = None,
#             global incumbent array, used when cs_local and cs_global have different number of dimensions and we need
# to
#             supplement the missing values.
#         """
#         self.cs_local = cs_local
#         self.challengers = challengers
#         self.cs_global = cs_global
#         self._index = 0
#         self.config_origin = config_origin
#         # In case cs_in and cs_out have different dimensions
#         self.expand_dims = len(cs_global.get_hyperparameters()) != len(cs_local.get_hyperparameters())
#         self.incumbent_array = incumbent_array

#         if self.expand_dims and self.incumbent_array is None:
#             raise ValueError(
#                 "Incumbent array must be provided if the global configuration space has more "
#                 "hyperparameters then the local configuration space"
#             )

#     def __next__(self) -> Configuration:
#         if self.challengers is not None and self._index == len(self.challengers):
#             raise StopIteration
#         challenger = self.challengers[self._index][1]
#         self._index += 1
#         value = challenger.get_dictionary()
#         if self.expand_dims:
#             incumbent_array = Configuration(
#                 configuration_space=self.cs_global, vector=self.incumbent_array
#             ).get_dictionary()
#             # we replace the cooresponding value in incumbent array with the value suggested by our optimizer
#             for k in value.keys():
#                 incumbent_array[k] = value[k]
#             config = Configuration(configuration_space=self.cs_global, values=incumbent_array)
#         else:
#             config = Configuration(configuration_space=self.cs_global, values=value)
#         if self.config_origin is not None:
#             config.origin = self.config_origin
#         else:
#             config.origin = challenger.origin
#         return config

#     def __len__(self) -> int:
#         if self.challengers is None:
#             self.challengers = []
#         return len(self.challengers) - self._index
