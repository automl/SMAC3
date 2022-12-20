from __future__ import annotations

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TS(AbstractAcquisitionFunction):
    r"""Thompson Sampling

    Warning
    -------
    Thompson Sampling can only be used together with `RandomSearch`. Please do not use `LocalAndSortedRandomSearch` to
    optimize the TS acquisition function!

    :math:`TS(X) ~ \mathcal{N}(\mu(\mathbf{X}),\sigma(\mathbf{X}))'
    Returns -TS(X) as the acquisition_function optimizer maximizes the acquisition value.

    Parameters
    ----------
    xi : float, defaults to 0.0
        TS does not require xi here, we only wants to make it consistent with other acquisition functions.
    """

    @property
    def name(self) -> str:  # noqa: D102
        return "Thompson Sampling"

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Sample a new value from a gaussian distribution whose mean and covariance values are given by model.

        Parameters
        ----------
        X: np.ndarray [N, D]
           Points to be evaluated where we could sample a value. N is the number of points and D the dimension
           for the points.

        Returns
        -------
        np.ndarray [N, 1]
            Negative sample value of X.
        """
        assert self._model

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        sample_function = getattr(self._model, "sample_functions", None)
        if callable(sample_function):
            return -sample_function(X, n_funcs=1)

        m, var_ = self._model.predict_marginalized(X)
        rng = self._model._rng
        m = m.flatten()
        var_ = np.diag(var_.flatten())

        return -rng.multivariate_normal(m, var_, 1).T
