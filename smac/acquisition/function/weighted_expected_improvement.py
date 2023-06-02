from __future__ import annotations

from typing import Any

import numpy as np
from scipy.stats import norm
from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

logger = get_logger(__name__)


class WEI(AbstractAcquisitionFunction):
    def __init__(self, alpha: float = 0.5, xi: float = 0, log: bool = False, use_pure_PI: bool = False) -> None:
        super().__init__()
        self._xi: float = xi
        self._log: bool = log
        if self._log:
            raise NotImplementedError
        self._eta: float | None = None
        self._alpha = alpha
        self._use_pure_PI = use_pure_PI

        self.pi_term: np.ndarray | None = None
        self.pi_pure_term: np.ndarray | None = None
        self.pi_mod_term: np.ndarray | None = None
        self.ei_term: np.ndarray | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Weighted Expected Improvement"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "xi": self._xi,
                "log": self._log,
                "alpha": self._alpha,
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
        alpha = kwargs.get("alpha", None)
        if alpha is not None:
            self._alpha = alpha

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
        if self._use_pure_PI:
            assert self._alpha == 1., f"{self._alpha} != 0.5 with use pure PI. Any other combination, especially alpha=0.5 (EI) leads to wrong WEI."

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if not self._log:
            if len(X.shape) == 1:
                X = X[:, np.newaxis]

            m, v = self._model.predict_marginalized(X)  # TODO: can the variance become negative?
            s = np.sqrt(v)

            def calculate_f() -> np.ndarray:
                z = (self._eta - m - self._xi) / s
                if self._use_pure_PI:
                    pi_term = norm.cdf(z)
                else:
                    pi_term = (self._eta - m - self._xi) * norm.cdf(z)
                ei_term = s * norm.pdf(z)
                self.pi_term = pi_term
                self.pi_pure_term = norm.cdf(z)
                self.pi_mod_term = (self._eta - m - self._xi) * norm.cdf(z)
                self.ei_term = ei_term
                return self._alpha * pi_term + (1 - self._alpha) * ei_term

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

            # if (f < 0).any():
            #     # TODO is it okay if this acq fun is smaller than 0?
            #     logger.warn("Expected Improvement is smaller than 0 for at least one " "sample.")
            #     # raise ValueError("Expected Improvement is smaller than 0 for at least one " "sample.")

            return f
        else:
            raise NotImplementedError
            # if len(X.shape) == 1:
            #     X = X[:, np.newaxis]

            # m, var_ = self._model.predict_marginalized(X)
            # std = np.sqrt(var_)

            # def calculate_log_ei() -> np.ndarray:
            #     # we expect that f_min is in log-space
            #     assert self._eta is not None
            #     assert self._xi is not None

            #     f_min = self._eta - self._xi
            #     v = (f_min - m) / std
            #     return self._alpha * (np.exp(f_min) * norm.cdf(v)) - (np.exp(0.5 * var_ + m) * norm.cdf(v - std))

            # if np.any(std == 0.0):
            #     # if std is zero, we have observed x on all instances
            #     # using a RF, std should be never exactly 0.0
            #     # Avoid zero division by setting all zeros in s to one.
            #     # Consider the corresponding results in f to be zero.
            #     logger.warning("Predicted std is 0.0 for at least one sample.")
            #     std_copy = np.copy(std)
            #     std[std_copy == 0.0] = 1.0
            #     log_ei = calculate_log_ei()
            #     log_ei[std_copy == 0.0] = 0.0
            # else:
            #     log_ei = calculate_log_ei()

            # if (log_ei < 0).any():
            #     raise ValueError("Expected Improvement is smaller than 0 for at least one sample.")

            return log_ei.reshape((-1, 1))


class EIPI(WEI):
    def __init__(self, alpha: float = 0.5, xi: float = 0, log: bool = False, use_pure_PI: bool = False) -> None:
        super().__init__(
            alpha=alpha,
            xi=xi,
            log=log,
            use_pure_PI=use_pure_PI
        )

    def _compute(self, X: np.ndarray) -> np.ndarray:
        match self._alpha:
            case 0.5:  # EI
                self._use_pure_PI = False
            case 1.:  # PI
                self._use_pure_PI = True
            case _:
                raise ValueError("Only values of alpha=0.5 -> EI and alpha=1 -> PI are valid.")
        return super()._compute(X=X)


