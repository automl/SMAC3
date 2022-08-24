from __future__ import annotations
from typing import Any

import numpy as np
from scipy.stats import norm

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.base_model import BaseModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PI(AbstractAcquisitionFunction):
    r"""Computes the probability of improvement for a given x over the best so far value as acquisition value.

    :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+}))` :math:`:= \Phi(\\frac{ \mu(\mathbf{X})-f(\mathbf{X^+}) }
    { \sigma(\mathbf{X}) })` with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal

    Parameters
    ----------
    par : float, default=0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.

    Attributes
    ----------
    long_name : str
    xi : float
        Exploration/exploitation trade-off parameter.
    eta : float
        Current incumbent value.
    """
    def __init__(self, xi: float = 0.0):
        super(PI, self).__init__()
        self.long_name : str= "Probability of Improvement"
        self.xi : float = xi
        self.eta : float | None = None

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def update(self, model: BaseModel, eta: float, xi : float | None = None, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        model : BaseModel
            Models the objective function.
        eta : float
            Current incumbent value.
        """
        self.model = model
        self.eta = eta
        if xi is not None:
            self.xi = xi

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the PI value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate PI. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<float>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)

        return norm.cdf((self.eta - m - self.xi) / std)
