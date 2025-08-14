# from __future__ import annotations

# from typing import Iterator

# from smac.acquisition.function.thompson import TS
# from ConfigSpace import Configuration
# from smac.main.smbo import SMBO
# from smac.model.utils import get_types
# from smac.utils.logging import get_logger
# from smac.utils.subspaces.turbo_subspace import TuRBOSubSpace


# logger = get_logger(__name__)


# class TuRBOSMBO(SMBO):
#     """
#     Interface to train the EPM and generate next configurations with TurBO:
#         D. Eriksson et al. Scalable Global Optimization via Local Bayesian Optimization
#         https://papers.nips.cc/paper/2019/file/6c990b7aca7bc7058f5e98ea909e924b-Paper.pdf

#     Parameters
#     ----------
#     length_init: float
#         Initialized length after restarting
#     length_min: float
#         If the subspace length is smaller than length_min, TurBO will restart
#     length_max: float
#         The maximum length of subspace
#     success_tol: int
#         Number of successful suggestions (suggested points become incumbent) required for expanding subspace
#     failure_tol_min: int
#         The minimum number of failure suggestions (suggested points fails to become incumbent) required for shrinking
#         subspace
#     n_init_x_params: int
#         how many configurations will be used at most in the initial design (X*D). Used for restarting the subspace
#     n_candidate_max: int
#         Maximal Number of points used as candidates
#     args:
#         additional arguments for initialize base SMBO object
#     kwargs:
#         additional arguments for initialize base SMBO object
#     """

#     def __init__(
#         self,
#         length_init: float = 0.8,
#         length_min: float = 0.5**8,
#         length_max: float = 1.6,
#         success_tol: int = 3,
#         failure_tol_min: int = 4,
#         n_init_x_params: int = 2,
#         n_candidate_max: int = 5000,
#         *args,
#         **kwargs,
#     ):
#         super(TuRBOSMBO, self).__init__(*args, **kwargs)

#         types, bounds = get_types(self.configspace, instance_features=None)

#         self.turbo = TuRBOSubSpace(
#             config_space=self.configspace,
#             bounds=bounds,
#             hps_types=types,
#             model_local=self.model,
#             acq_func_local=TS,
#             length_init=length_init,
#             length_min=length_min,
#             length_max=length_max,
#             success_tol=success_tol,
#             failure_tol_min=failure_tol_min,
#             n_init_x_params=n_init_x_params,
#             n_candidate_max=n_candidate_max,
#         )

#     def ask(self) -> Iterator[Configuration]:
#         """
#         Choose next candidate solution with TuRBO

#         Parameters
#         ----------
#         incumbent_value: float
#             Cost value of incumbent configuration (required for acquisition function);
#             If not given, it will be inferred from runhistory or predicted;
#             if not given and runhistory is empty, it will raise a ValueError.

#         Returns
#         -------
#         Iterator
#         """
#         logger.debug("Search for next configuration")
#         X, Y, X_configurations = self._collect_data()

#         num_new_bservations = 1  # here we only consider batch size = 1

#         new_observations = Y[-num_new_bservations:]
#         if len(self.turbo.init_configs) > 0:
#             self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])
#             return self.turbo.generate_challengers()
#         self.turbo.adjust_length(new_observations)

#         self.turbo.add_new_observations(X[-num_new_bservations:], Y[-num_new_bservations:])

#         challengers = self.turbo.generate_challengers()
#         return challengers
