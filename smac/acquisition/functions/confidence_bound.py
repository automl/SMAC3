from __future__ import annotations
from typing import Any

import numpy as np

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

class LCB(AbstractAcquisitionFunction):
    r"""Computes the lower confidence bound for a given x over the best so far value as
    acquisition value.

    :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})` [1]_

    with

    :math:`\beta_t = 2 \log( |D| t^2 / \beta)`

    :math:`\text{Input space} D`
    :math:`\text{Number of input dimensions} |D|`
    :math:`\text{Number of data points} t`
    :math:`\text{Exploration/exploitation tradeoff} \beta`

    Returns -LCB(X) as the acquisition_function optimizer maximizes the acquisition value.

    References
    ----------
    .. [1] [GP-UCB](https://arxiv.org/pdf/0912.3995.pdf)

    Parameters
    ----------
    model : BaseEPM
        A model that implements at least
                - predict_marginalized_over_instances(X)
    beta : float, defaults to 1.0
        Controls the balance between exploration and exploitation of the
        acquisition function.

    Attributes
    ----------
    long_name : str 
    beta : float
        Exploration / exploitation trade-off parameter.
    num_data : int | None
        Number of data points (t).
    """
    def __init__(self, beta: float = 1.0):
        super(LCB, self).__init__()
        self.long_name : str = "Lower Confidence Bound"
        self.beta : float = beta
        self.num_data : int | None = None
        self._required_updates : tuple(str, ...) = ("model", "num_data")

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the LCB value.

        Parameters
        ----------
        X: np.ndarray(N, D)
           Points to evaluate LCB. N is the number of points and D the dimension for the points

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self.num_data is None:
            raise ValueError(
                "No current number of Datapoints specified. Call update("
                "num_data=<int>) to inform the acquisition function "
                "about the number of datapoints."
            )
        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self.model.predict_marginalized_over_instances(X)
        std = np.sqrt(var_)
        beta_t = 2 * np.log((X.shape[1] * self.num_data**2) / self.beta)
        return -(m - np.sqrt(beta_t) * std)
