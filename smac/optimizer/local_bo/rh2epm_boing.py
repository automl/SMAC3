import typing

import copy

import numpy as np

from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import (
    RunHistory2EPM4Cost,
    RunHistory2EPM4LogScaledCost,
)


class RunHistory2EPM4CostWithRaw(RunHistory2EPM4Cost):
    """
    A transformer that transform RUnHistroy to vectors, Here addition to the transformed values, we will also
    return the raw values
    """

    def transform_with_raw(
        self,
        runhistory: RunHistory,
        budget_subset: typing.Optional[typing.List] = None,
    ) -> typing.Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Returns vector representation of runhistory; if imputation is
        disabled, censored (TIMEOUT with time < cutoff) will be skipped

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
        X, Y_raw = RunHistory2EPM4Cost.transform(self, runhistory, budget_subset)
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
        """Transform function response values. Returns the input values before transformation

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values


class RunHistory2EPM4ScaledLogCostWithRaw(RunHistory2EPM4CostWithRaw, RunHistory2EPM4LogScaledCost):
    def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values. Returns the input values before transformation

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return RunHistory2EPM4LogScaledCost.transform_response_values(self, values)
