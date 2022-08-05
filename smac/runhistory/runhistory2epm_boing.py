from __future__ import annotations

import typing

import copy

import numpy as np

from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory_transformer import (
    RunhistoryLogScaledTransformer,
    RunhistoryTransformer,
)


class RunHistory2EPM4CostWithRaw(RunhistoryTransformer):
    """
    A transformer that transform RunHistroy to vectors, this set of classes will return the raw cost values in
    addition to the transformed cost values. The raw cost values can then be applied for local BO approaches.
    """

    def transform_with_raw(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns vector representation of runhistory; if imputation is
        disabled, censored (TIMEOUT with time < cutoff) will be skipped. This function returns both the raw
        and transformed cost values

        Parameters
        ----------
        runhistory : smac.runhistory.runhistory.RunHistory
            Runhistory containing all evaluated configurations/instances
        budget_subset : list of budgets to consider

        Returns
        -------
        X: numpy.ndarray
            configuration vector x instance features
        Y: numpy.ndarray
            cost values
        Y_raw: numpy.ndarray
            cost values before transformation
        """
        X, Y_raw = RunhistoryTransformer.transform(self, runhistory, budget_subset)
        Y = copy.deepcopy(Y_raw)
        Y = self.transform_raw_values(Y)
        return X, Y, Y_raw

    def transform_response_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Returns the input values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        # otherwise it will be overwritten by its superclass
        return values

    def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Returns the raw input values before transformation

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values


class RunHistory2EPM4ScaledLogCostWithRaw(RunHistory2EPM4CostWithRaw, RunhistoryLogScaledTransformer):
    def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Returns the raw input values before transformation

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return RunhistoryLogScaledTransformer.transform_response_values(self, values)
