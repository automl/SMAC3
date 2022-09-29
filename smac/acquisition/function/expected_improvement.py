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


class EI(AbstractAcquisitionFunction):
    r"""Computes the expected improvement for a given x.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.
    log : bool, defaults to False
        Whether the function values are in log-space.

    """

    def __init__(
        self,
        xi: float = 0.0,
        log: bool = False,
    ) -> None:
        super(EI, self).__init__()

        self._xi: float = xi
        self._log: bool = log
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
            }
        )

        return meta

    def _update(self, **kwargs: Any) -> None:
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computess the EI value and its derivatives."""
        assert self._model is not None
        assert self._xi is not None
        if self._eta is None:
            raise ValueError("No `eta` specified. Call `update` to inform the acqusition function.")

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)
            s = np.sqrt(v)

            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                return (self._eta - m - self._xi) * norm.cdf(z) + s * norm.pdf(z)

            if np.any(s == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                s_copy = np.copy(s)
                s[s_copy == 0.0] = 1.0
                f = calculate_f()
                f[s_copy == 0.0] = 0.0
            else:
                f = calculate_f()
            if (f < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

            return f
        else:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, var_ = self._model.predict_marginalized(X)
            std = np.sqrt(var_)

            def calculate_log_ei() -> np.ndarray:
                # we expect that f_min is in log-space
                assert self._eta is not None
                assert self._xi is not None

                f_min = self._eta - self._xi
                v = (f_min - m) / std
                return (np.exp(f_min) * norm.cdf(v)) - (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

            if np.any(std == 0.0):
                # if std is zero, we have observed x on all instances
                # using a RF, std should be never exactly 0.0
                # Avoid zero division by setting all zeros in s to one.
                # Consider the corresponding results in f to be zero.
                logger.warning("Predicted std is 0.0 for at least one sample.")
                std_copy = np.copy(std)
                std[std_copy == 0.0] = 1.0
                log_ei = calculate_log_ei()
                log_ei[std_copy == 0.0] = 0.0
            else:
                log_ei = calculate_log_ei()

            if (log_ei < 0).any():
                raise ValueError("Expected Improvement is smaller than 0 for at least one sample.")

            return log_ei.reshape((-1, 1))


class EIPS(EI):
    r"""Computess for a given x the expected improvement as acquisition value.

    :math:`EI(X) := \frac{\mathbb{E}\left[\max\{0,f(\mathbf{X^+})-f_{t+1}(\mathbf{X})-\xi\right]\}]}{np.log(r(x))}`,
    with :math:`f(X^+)` as the best location and :math:`r(x)` as runtime.

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the acquisition function.
    """

    def __init__(self, xi: float = 0.0) -> None:
        super(EIPS, self).__init__(xi=xi)

    @property
    def name(self) -> str:  # noqa: D102
        return "Expected Improvement per Second"

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computess the EIPS value."""
        assert self._model is not None
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self._model.predict_marginalized(X)
        if m.shape[1] != 2:
            raise ValueError("m has wrong shape: %s != (-1, 2)" % str(m.shape))
        if v.shape[1] != 2:
            raise ValueError("v has wrong shape: %s != (-1, 2)" % str(v.shape))

        m_cost = m[:, 0]
        v_cost = v[:, 0]

        # The model already predicts log(runtime)
        m_runtime = m[:, 1]
        s = np.sqrt(v_cost)

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        def calculate_f() -> np.ndarray:
            z = (self._eta - m_cost - self._xi) / s
            f = (self._eta - m_cost - self._xi) * norm.cdf(z) + s * norm.pdf(z)
            f = f / m_runtime

            return f

        if np.any(s == 0.0):
            # if std is zero, we have observed x on all instances
            # using a RF, std should be never exactly 0.0
            # Avoid zero division by setting all zeros in s to one.
            # Consider the corresponding results in f to be zero.
            logger.warning("Predicted std is 0.0 for at least one sample.")
            s_copy = np.copy(s)
            s[s_copy == 0.0] = 1.0
            f = calculate_f()
            f[s_copy == 0.0] = 0.0
        else:
            f = calculate_f()

        if (f < 0).any():
            raise ValueError("Expected Improvement per Second is smaller than 0 " "for at least one sample.")

        return f.reshape((-1, 1))
