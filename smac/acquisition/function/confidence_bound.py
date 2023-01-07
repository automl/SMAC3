from __future__ import annotations

from typing import Any

import numpy as np

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class LCB(AbstractAcquisitionFunction):
    r"""Computes the lower confidence bound for a given x over the best so far value as acquisition value.

    :math:`LCB(X) = \mu(\mathbf{X}) - \sqrt(\beta_t)\sigma(\mathbf{X})` [SKKS10]_

    with

    :math:`\beta_t = 2 \log( |D| t^2 / \beta)`

    :math:`\text{Input space} D`
    :math:`\text{Number of input dimensions} |D|`
    :math:`\text{Number of data points} t`
    :math:`\text{Exploration/exploitation tradeoff} \beta`

    Returns -LCB(X) as the acquisition_function optimizer maximizes the acquisition value.

    Parameters
    ----------
    beta : float, defaults to 1.0
        Controls the balance between exploration and exploitation of the acquisition function.

    Attributes
    ----------
    _beta : float
        Exploration-exploitation trade-off parameter.
    _num_data : int
        Number of data points seen so far.
    """

    def __init__(self, beta: float = 1.0) -> None:
        super(LCB, self).__init__()
        self._beta: float = beta
        self._num_data: int | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Lower Confidence Bound"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"beta": self._beta})

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update acsquisition function attributes

        Parameters
        ----------
        num_data : int
            Number of data points
        """
        assert "num_data" in kwargs
        self._num_data = kwargs["num_data"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute LCB acquisition value

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X.

        Raises
        ------
        ValueError
            If `update` has not been called before. Number of data points is unspecified in this case.
        """
        assert self._model is not None
        if self._num_data is None:
            raise ValueError(
                "No current number of data points specified. Call `update` to inform the acqusition function."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, var_ = self._model.predict_marginalized(X)
        std = np.sqrt(var_)
        beta_t = 2 * np.log((X.shape[1] * self._num_data**2) / self._beta)

        return -(m - np.sqrt(beta_t) * std)
