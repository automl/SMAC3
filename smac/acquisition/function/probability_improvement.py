from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PI(AbstractAcquisitionFunction):
    r"""Probability of Improvement

    :math:`P(f_{t+1}(\mathbf{X})\geq f(\mathbf{X^+}))` :math:`:= \Phi(\\frac{ \mu(\mathbf{X})-f(\mathbf{X^+}) }
    { \sigma(\mathbf{X}) })` with :math:`f(X^+)` as the incumbent and :math:`\Phi` the cdf of the standard normal.

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the acquisition function.
    """

    def __init__(self, xi: float = 0.0):
        super(PI, self).__init__()
        self._xi: float = xi
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Probability of Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"xi": self._xi})

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update acsquisition function attributes

        Parameters
        ----------
        eta : float
            Function value of current incumbent.
        xi : float, optional
            Exploration-exploitation trade-off parameter
        """
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the PI value.

        Parameters
        ----------
        X: np.ndarray [N, D]
           Points to evaluate PI. N is the number of points and D the dimension for the points.

        Returns
        -------
        np.ndarray [N, 1]
            Expected Improvement of X.

        Raises
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).

        """
        assert self._model is not None
        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<float>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]
        m, var_ = self._model.predict_marginalized(X)
        std = np.sqrt(var_)

        return norm.cdf((self._eta - m - self._xi) / std)
