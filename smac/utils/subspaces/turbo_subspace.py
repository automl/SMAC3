# from __future__ import annotations

# from typing import Dict, List, Optional, Tuple, Type, Union

# import math
# import warnings

# import numpy as np
# from ConfigSpace.hyperparameters import NumericalHyperparameter
# from ConfigSpace.util import deactivate_inactive_hyperparameters
# from scipy.stats.qmc import LatinHypercube, Sobol

# from smac.acquisition import AbstractAcquisitionMaximizer
# from smac.acquisition.function import AbstractAcquisitionFunction, TS
# from ConfigSpace import Configuration, ConfigurationSpace
# from smac.model.abstract_model import AbstractModel
# from smac.model.gaussian_process.gpytorch_gaussian_process import GloballyAugmentedLocalGaussianProcess
# from smac.model.gaussian_process.abstract_gaussian_process import AbstractGaussianProcess
# from smac.model.gaussian_process.gpytorch_gaussian_process import GPyTorchGaussianProcess
# from smac.model.gaussian_process.mcmc_gaussian_process import MCMCGaussianProcess
# from smac.utils.logging import get_logger
# from smac.utils.subspaces import LocalSubspace

# logger = get_logger(__name__)

# warnings.filterwarnings("ignore", message="The balance properties of Sobol' points require" " n to be a power of 2.")


# class TuRBOSubSpace(LocalSubspace):
#     """
#     Subspace designed for TurBO:
#         D. Eriksson et al. Scalable Global Optimization via Local Bayesian Optimization
#         https://proceedings.neurips.cc/paper/2019/hash/6c990b7aca7bc7058f5e98ea909e924b-Abstract.html

#     The hyperparameters follow the illustration under supplementary D, `TuRBO details`.

#     Parameters
#     ----------
#     length_init: float
#         initialized length of subspace
#     length_min: float
#         the minimal length of subspace, if the subspace has a length smaller than this value, turbo will restart
#     length_max: float
#         the maximal length of subspace
#     success_tol: float
#        the number of successive successful evaluations required for expanding the subregion
#     failure_tol_min: float
#        the minimal number of successive successful evaluations required for shrinking the subregion (otherwise
#        this value is set as number of feature dimensions)
#     n_init_x_params: int
#         how many configurations will be used at most in the initial design (X*D). Used for restarting the subspace
#     n_candidate_max: int
#         The maximal Number of points used as candidates
#     """

#     def __init__(
#         self,
#         config_space: ConfigurationSpace,
#         bounds: List[Tuple[float, float]],
#         hps_types: List[int],
#         bounds_ss_cont: Optional[np.ndarray] = None,
#         bounds_ss_cat: Optional[List[Tuple]] = None,
#         model_local: Union[AbstractModel, Type[AbstractModel]] = GPyTorchGaussianProcess,
#         model_local_kwargs: Dict = {},
#         acq_func_local: Union[AbstractAcquisitionFunction, Type[AbstractAcquisitionFunction]] = TS,
#         acq_func_local_kwargs: Optional[Dict] = None,
#         rng: Optional[np.random.RandomState] = None,
#         initial_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
#         activate_dims: Optional[np.ndarray] = None,
#         incumbent_array: Optional[np.ndarray] = None,
#         length_init: float = 0.8,
#         length_min: float = 0.5**7,
#         length_max: float = 1.6,
#         success_tol: int = 3,
#         failure_tol_min: int = 4,
#         n_init_x_params: int = 2,
#         n_candidate_max: int = 5000,
#     ):
#         self.num_valid_observations = 0
#         super(TuRBOSubSpace, self).__init__(
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
#         hps = config_space.get_hyperparameters()
#         for hp in hps:
#             if not isinstance(hp, NumericalHyperparameter):
#                 raise ValueError("Current TurBO Optimizer only supports Numerical Hyperparameters")
#         if len(config_space.get_conditions()) > 0 or len(config_space.get_forbiddens()) > 0:
#             raise ValueError("Currently TurBO does not support Conditional or Forbidden Hyperparameters")

#         n_hps = len(self.activate_dims)
#         self.n_dims = n_hps
#         self.n_init = n_init_x_params * self.n_dims
#         self.n_candidates = min(100 * n_hps, n_candidate_max)

#         self.failure_tol = max(failure_tol_min, n_hps)
#         self.success_tol = success_tol
#         self.length = length_init
#         self.length_init = length_init
#         self.length_min = length_min
#         self.length_max = length_max
#         self._restart_turbo(n_init_points=self.n_init)

#         if initial_data is not None:
#             self.add_new_observations(initial_data[0], initial_data[1])
#             self.init_configs = []  # type: List[Configuration]

#         self.lb = np.zeros(self.n_dims)
#         self.ub = np.ones(self.n_dims)
#         self.config_origin = "TuRBO"

#     def _restart_turbo(
#         self,
#         n_init_points: int,
#     ) -> None:
#         """
#         Restart TurBO with a certain number of initialized points. New points are initialized with latin hypercube

#         Parameters
#         ----------
#         n_init_points: int
#             number of points required for initializing a new subspace
#         """
#         logger.debug("Current length is smaller than the minimal value, a new TuRBO restarts")
#         self.success_count = 0
#         self.failure_count = 0

#         self.num_eval_this_round = 0
#         self.last_incumbent_value = np.inf
#         self.length = self.length_init

#         self.num_valid_observations = 0

#         init_vectors = LatinHypercube(d=self.n_dims, seed=np.random.seed(self.rng.randint(1, 2**20))).random(
#             n=n_init_points
#         )

#         self.init_configs = [Configuration(self.cs_local, vector=init_vector) for init_vector in init_vectors]

#     def adjust_length(self, new_observation: Union[float, np.ndarray]) -> None:
#         """
#         Adjust the subspace length according to the performance of the latest suggested values
#         Parameters
#         ----------
#         new_observation: Union[float, np.ndarray]
#             new observations
#         """
#         # see Section 2: 'Trust regions'
#         optim_observation = new_observation if np.isscalar(new_observation) else np.min(new_observation)

#         # We define a ``success'' as a candidate that improves upon $\xbest$, and a ``failure'' as a candidate that
#         # does not.
#         if optim_observation < np.min(self.model_y) - 1e-3 * math.fabs(np.min(self.model_y)):
#             logger.debug("New suggested value is better than the incumbent, success_count increases")
#             self.success_count += 1
#             self.failure_count = 0
#         else:
#             logger.debug("New suggested value is worse than the incumbent, failure_count increases")
#             self.success_count = 0
#             self.failure_count += 1

#         # After $\tau_{\text{succ}}$ consecutive successes, we double the size of the TR,
#         # i.e., $\len \gets \min\{\len_{\textrm{max}}, 2\len\}$.
#         if self.success_count == self.success_tol:  # Expand trust region
#             self.length = min([2.0 * self.length, self.length_max])
#             self.success_count = 0
#             logger.debug(f"Subspace length expands to {self.length}")
#         # After $\tau_{\text{fail}}$ consecutive failures, we halve the size of the TR: $\len \gets \len/2$.
#         # We reset the success and failure counters to zero after we change the size of the TR.
#         elif self.failure_count == self.failure_tol:  # Shrink trust region
#             self.length /= 2.0
#             self.failure_count = 0
#             logger.debug(f"Subspace length shrinks to {self.length}")

#     def _generate_challengers(  # type: ignore[override]
#         self, _sorted: bool = True
#     ) -> List[Tuple[float, Configuration]]:
#         """
#         Generate new challengers list for this subspace

#         Parameters
#         ----------
#         _sorted: bool
#             if the generated challengers are sorted by their acquisition function values
#         """
#         if len(self.init_configs) > 0:
#             config_next = self.init_configs.pop()
#             return [(0, config_next)]

#         if self.length < self.length_min:
#             self._restart_turbo(n_init_points=self.n_init)
#             config_next = self.init_configs.pop()
#             return [(0, config_next)]

#         self.model.train(self.model_x[-self.num_valid_observations :], self.model_y[-self.num_valid_observations :])
#         self.update_model(predict_x_best=False, update_incumbent_array=True)

#         sobol_gen = Sobol(d=self.n_dims, scramble=True, seed=self.rng.randint(low=0, high=10000000))
#         sobol_seq = sobol_gen.random(self.n_candidates)

#         # adjust length according to kernel length
#         if isinstance(
#             self.model,
#             (
#                 AbstractGaussianProcess,
#                 MCMCGaussianProcess,
#                 GloballyAugmentedLocalGaussianProcess,
#                 GPyTorchGaussianProcess,
#             ),
#         ):
#             if isinstance(self.model, AbstractGaussianProcess):
#                 kernel_length = np.exp(self.model.hypers[1:-1])
#             elif isinstance(self.model, MCMCGaussianProcess):
#                 kernel_length = np.exp(np.mean((np.array(self.model.hypers)[:, 1:-1]), axis=0))
#             elif isinstance(self.model, (GPyTorchGaussianProcess, GloballyAugmentedLocalGaussianProcess)):
#                 kernel_length = self.model.kernel.base_kernel.lengthscale.cpu().detach().numpy()

#             # See section 'Trust regions' of section 2
#             #  $\len_i = \lambda_i L / (\prod_{j=1}^d \lambda_j)^{1/d}$,
#             # We now have weights.prod() = 1
#             # This makes the result more stable
#             subspace_scale = kernel_length / np.prod(np.power(kernel_length, 1.0 / self.n_dims))

#             subspace_length = self.length * subspace_scale

#             subspace_lb = np.clip(self.incumbent_array - subspace_length * 0.5, 0.0, 1.0)
#             subspace_ub = np.clip(self.incumbent_array + subspace_length * 0.5, 0.0, 1.0)
#             sobol_seq = sobol_seq * (subspace_ub - subspace_lb) + subspace_lb

#         prob_perturb = min(20.0 / self.n_dims, 1.0)
#         design = self._perturb_samples(prob_perturb, sobol_seq)

#         # Only numerical hyperpameters are considered for TuRBO, we don't need to transfer the vectors to fit the
#         # requirements of other sorts of hyperparameters
#         configs = []
#         for vector in design:
#             conf = deactivate_inactive_hyperparameters(
#                 configuration=None, configuration_space=self.cs_local, vector=vector
#             )
#             configs.append(conf)

#         if _sorted:
#             return AbstractAcquisitionMaximizer._sort_configs_by_acq_value(self, configs)
#         else:
#             return [(0, configs[i]) for i in range(len(configs))]

#     def _perturb_samples(self, prob_perturb: float, design: np.ndarray) -> np.ndarray:
#         """
#         See Supplementary D, 'TuRBO details':
#         In order to not perturb all coordinates at once, we use the value in the Sobol sequence
#         with probability min{1,20/d} for a given candidate and dimension, and the value of the center otherwise

#         perturb the generated design with the incumbent array accordingly

#         Parameters
#         ----------
#         prob_perturb: float
#             probability that a design is perturbed by the incumbent value
#         design: np.ndarray(self.n_candidates, self.n_dims)
#             design array to be perturbed
#         Returns
#         -------
#         design_perturbed: np.ndarray(self.n_candidates, self.n_dims)
#             perturbed design array
#         """
#         # we will use masked array, thus the indices that will be replaced will be marked with True
#         mask = self.rng.rand(self.n_candidates, self.n_dims) > prob_perturb

#         ind = np.where(np.sum(mask, axis=1) == self.n_dims)[0]
#         if self.n_dims == 1:
#             mask[ind, 0] = 0
#         else:
#             # ensure that no candidate will be completely replaced by the incumbent value
#             mask[ind, self.rng.randint(0, self.n_dims, size=len(ind))] = 0
#         return np.ma.array(design, mask=mask, fill_value=self.incumbent_array).filled()

#     def add_new_observations(self, X: np.ndarray, y: np.ndarray) -> None:
#         """
#         Add new observations to the subspace, meanwhile, we add the number of valid observation to ensure that the
#         subspace could be scaled properly.

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
#         super(TuRBOSubSpace, self).add_new_observations(X, y)
#         self.num_valid_observations += len(y)
