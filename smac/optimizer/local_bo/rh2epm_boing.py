import numpy as np
import typing
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost

from smac.runhistory.runhistory import RunHistory
from smac.utils import constants


class RunHistory2EPM4CostWithRaw(RunHistory2EPM4Cost):
    """
    A transformer that transform RUnHistroy to vectors, Here addition to the transformed values, we will also
    return the raw values
    """

    def transform(
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
        """
        X, Y_raw = super(RunHistory2EPM4CostWithRaw, self).transform(runhistory, budget_subset)
        Y = self.transform_raw_values(Y_raw)
        return X, Y, Y_raw

    def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function raw values.

        Returns the input values.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        return values


class RunHistory2EPM4LogCostWithRaw(RunHistory2EPM4CostWithRaw):
    def transform_raw_values(self, values: np.ndarray) -> np.ndarray:
        """Transform function response values.

        Transforms the response values by using a log transformation.

        Parameters
        ----------
        values : np.ndarray
            Response values to be transformed.

        Returns
        -------
        np.ndarray
        """
        # ensure that minimal value is larger than 0
        min_y = self.min_y - (self.perc - self.min_y)  # Subtract the difference between the percentile and the minimum
        min_y -= constants.VERY_SMALL_NUMBER  # Minimal value to avoid numerical issues in the log scaling below
        # linear scaling
        if min_y == self.max_y:
            # prevent diving by zero
            min_y *= 1 - 10 ** -10
        values_return = (values - min_y) / (self.max_y - min_y)
        values_return = np.log(values_return)
        return values_return
