from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class EI(AbstractAcquisitionFunction):
    r"""The Expected Improvement (EI) criterion is used to decide where to evaluate a function f(x) next. The goal is to
    balance exploration and exploitation. Expected Improvement (with or without function values in log space)
    acquisition function

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.

    Reference for EI: Jones, D.R. and Schonlau, M. and Welch, W.J. (1998). Efficient Global Optimization of Expensive
    Black-Box Functions. Journal of Global Optimization 13, 455–492

    Reference for logEI: Hutter, F. and Hoos, H. and Leyton-Brown, K. and Murphy, K. (2009). An experimental
    investigation of model-based parameter optimisation: SPO and beyond. In: Conference on Genetic and
    Evolutionary Computation

    The logEI implemententation is based on the derivation of the orginal equation by:
    Watanabe, S. (2024). Derivation of Closed Form of Expected Improvement for Gaussian Process Trained on
    Log-Transformed Objective. https://arxiv.org/abs/2411.18095

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.
    log : bool, defaults to False
        Whether the function values are in log-space.


    Attributes
    ----------
    _xi : float
        Exploration-exloitation trade-off parameter.
    _log: bool
        Function values in log-space or not.
    _eta : float
        Current incumbent function value (best value observed so far).

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
        """Compute EI acquisition value

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
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EI is < 0 for at least one sample (normal function value space).
        ValueError
            If EI is < 0 for at least one sample (log function value space).
        """
        assert self._model is not None
        assert self._xi is not None

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

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
    r"""Expected Improvement per Second acquisition function

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
        """Compute EI per second acquisition value

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
            If the mean has the wrong shape, should have shape (-1, 2).
        ValueError
            If the variance has the wrong shape, should have shape (-1, 2).
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        ValueError
            If EIPS is < 0 for at least one sample.
        """
        assert self._model is not None
        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        m, v = self._model.predict_marginalized(X)
        if m.shape[1] != 2:
            raise ValueError(f"m has wrong shape: {m.shape} != (-1, 2)")
        if v.shape[1] != 2:
            raise ValueError(f"v has wrong shape: {v.shape} != (-1, 2)")

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


class QExpectedImprovement(EI):
    r"""
    Monte Carlo approximation of q-Expected Improvement.
    Approximates joint distribution with independent normals.

    :math:`EI(X) := \mathbb{E}\left[ \max\{0, f(\mathbf{X^+}) - f_{t+1}(\mathbf{X}) - \xi \} \right]`,
    with :math:`f(X^+)` as the best location.

    Reference for q-EI


    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation of the
        acquisition function.
    log : bool, defaults to False
        Whether the function values are in log-space.


    Attributes
    ----------
    _xi : float
        Exploration-exloitation trade-off parameter.
    _log: bool
        Function values in log-space or not.
    _eta : float
        Current incumbent function value (best value observed so far).

    """

    def __init__(self, xi: float = 0.0, n_samples: int = 128) -> None:
        super(QExpectedImprovement, self).__init__(xi=xi)
        self.n_samples = n_samples

    @property
    def name(self) -> str:  # noqa: D102
        return "Batch Expected Improvement"

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """
        Compute q-EI acquisition value using Monte Carlo approximation.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The batch of input points to evaluate.

        Returns
        -------
        np.ndarray [1, 1]
            The q-EI value for the batch as a whole.
        """
        assert self._model is not None
        assert self._xi is not None

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update(eta=<float>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        m, var = self._model.predict_marginalized(X)
        std = np.sqrt(var)

        if np.any(std == 0.0):
            logger.warning("Predicted std is 0.0 for at least one sample.")
            std_copy = np.copy(std)
            std[std_copy == 0.0] = 1.0  # prevent division by zero

        # Monte Carlo sampling from log-normal distribution
        normal_samples = np.random.normal(loc=m.T, scale=std.T, size=(self.n_samples, X.shape[0]))

        if not self._log:
            f_samples = normal_samples  # in original (normal) space
            f_min_sample = np.min(f_samples, axis=1)
            improvement = np.maximum(self._eta - self._xi - f_min_sample, 0.0)
        else:
            # In log-space, the *actual values* are exp(samples)
            f_samples = np.exp(normal_samples)
            f_min_sample = np.min(f_samples, axis=1)

            # eta is already in log-space, so we compare to exp(eta - xi)
            improvement = np.maximum(np.exp(self._eta - self._xi) - f_min_sample, 0.0)

        qei = np.mean(improvement)

        if qei < 0:
            raise ValueError("q-Expected Improvement is smaller than 0. Should not happen.")

        return np.array([[qei]])
