# from __future__ import annotations

# import copy

# import numpy as np

# from smac.runhistory.encoder.encoder import RunHistoryEncoder
# from smac.runhistory.encoder.log_scaled_encoder import RunHistoryLogScaledEncoder
# from smac.runhistory.runhistory import RunHistory
# from smac.utils.logging import get_logger

# __copyright__ = "Copyright 2022, automl.org"
# __license__ = "3-clause BSD"


# logger = get_logger(__name__)


# class RunHistoryRawEncoder(RunHistoryEncoder):
#     """
#     A transformer that transform the RunHistroy to vectors. This set of classes will return the raw cost values in
#     addition to the transformed cost values. The raw cost values can then be applied for local BO approaches.
#     """

#     def transform_with_raw(
#         self,
#         runhistory: RunHistory,
#         budget_subset: list | None = None,
#     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
#         """Returns vector representation of runhistory; if imputation is
#         disabled, censored (TIMEOUT with time < cutoff) will be skipped. This function returns both the raw
#         and transformed cost values

#         Parameters
#         ----------
#         runhistory : smac.runhistory.runhistory.RunHistory
#             Runhistory containing all evaluated configurations/instances
#         budget_subset : list of budgets to consider

#         Returns
#         -------
#         X: numpy.ndarray
#             configuration vector x instance features
#         Y: numpy.ndarray
#             cost values
#         Y_raw: numpy.ndarray
#             cost values before transformation
#         """
#         X, Y_raw = RunHistoryEncoder.transform(self, runhistory, budget_subset)
#         Y = copy.deepcopy(Y_raw)
#         Y = self.transform_raw_values(Y)
#         return X, Y, Y_raw

#     def transform_response_values(self, values: np.ndarray) -> np.ndarray:
#         """Returns the input values."""
#         return values

#     def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
#         """Returns the raw input values before transformation."""
#         return values


# class RunHistoryRawScaledEncoder(RunHistoryRawEncoder, RunHistoryLogScaledEncoder):
#     def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
#         """Returns the raw input values before transformation."""
#         return RunHistoryLogScaledEncoder.transform_response_values(self, values)
