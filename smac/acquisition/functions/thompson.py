from __future__ import annotations
from typing import Any

import numpy as np

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class TS(AbstractAcquisitionFunction):
    r"""Do a Thompson Sampling for a given x over the best so far value as
    acquisition value.

    Warning
    -------
    Thompson Sampling can only be used together with
    smac.acquisition.random_search.RandomSearch, please do not use
    smac.acquisition.local_and_random_search.LocalAndSortedRandomSearch to optimize TS
    acquisition function!

    :math:`TS(X) ~ \mathcal{N}(\mu(\mathbf{X}),\sigma(\mathbf{X}))'
    Returns -TS(X) as the acquisition_function optimizer maximizes the acquisition value.

    Parameters
    ----------
    par : float, defaults to 0.0
        TS does not require par here, we only wants to make it consistent with
        other acquisition functions.

    Attributes
    ----------
    long_name : str
    par : float
        Exploration/exploitation trade-off parameter.
    num_data : int
        Number of data points (t).
    """

    def __init__(self, par: float = 0.0) -> None:
        # TODO check if TS is used with RandomSearch only
        super(TS, self).__init__()
        self.long_name: str = "Thompson Sampling"
        self.par: float = par
        self.num_data: int | None = None

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def update(self, model: AbstractModel, num_data: int, par: float | None = None, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        model : BaseModel
            Models the objective function.
        """
        self.model = model
        self.num_data = num_data
        if par is not None:
            self.par = par

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Sample a new value from a gaussian distribution whose mean and covariance values
        are given by model.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to be evaluated where we could sample a value. N is the number of points and D the dimension
           for the points

        Returns
        -------
        np.ndarray(N,1)
            negative sample value of X
        """
        assert self.model

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        sample_function = getattr(self.model, "sample_functions", None)
        if callable(sample_function):
            return -sample_function(X, n_funcs=1)

        m, var_ = self.model.predict_marginalized_over_instances(X)
        rng = getattr(self.model, "rng", np.random.RandomState(self.model.seed))
        m = m.flatten()
        var_ = np.diag(var_.flatten())
        return -rng.multivariate_normal(m, var_, 1).T
