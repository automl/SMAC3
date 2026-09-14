from __future__ import annotations

import math
from typing import Any

import numpy as np
from scipy.special import erfcx, log_ndtr
from scipy.stats import norm

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

_LOG_SQRT_2PI = 0.5 * math.log(2.0 * math.pi)
_LOG_SQRT_PI_2 = 0.5 * math.log(math.pi / 2.0)
_LOG_2 = math.log(2.0)

# Below this the series expansion of erfcx has converged to within floating point precision, so the closed form
# collapses to its asymptote and evaluating it directly would only lose digits.
_ASYMPTOTIC_THRESHOLD = -1.0 / math.sqrt(np.finfo(float).eps)


def _log1mexp(x: np.ndarray) -> np.ndarray:
    """Computes ``log(1 - exp(x))`` for negative ``x`` without cancellation."""
    return np.where(x > -_LOG_2, np.log(-np.expm1(x)), np.log1p(-np.exp(x)))


def log_h(z: np.ndarray) -> np.ndarray:
    r"""Logarithm of $h(z) = \phi(z) + z \Phi(z)$, the standardised expected improvement.

    Evaluating $h$ and then taking the logarithm loses the value entirely once $z$ is a few standard
    deviations negative, because $h$ underflows to zero while its logarithm is perfectly representable. That
    region is most of the search space late in a run, which is why expected improvement becomes hard to
    maximize. See [[ADE+23][ADE+23]] for the derivation.

    For $z \le -1$ the identity $\Phi(z) = \tfrac{1}{2}\mathrm{erfcx}(-z/\sqrt{2})e^{-z^2/2}$ pulls the
    exponential out front, leaving

    $$
    \log h(z) = -\frac{z^2}{2} - \log\sqrt{2\pi}
                + \log\!\left(1 - |z|\sqrt{\pi/2}\,\mathrm{erfcx}(|z|/\sqrt{2})\right),
    $$

    whose last term is computed with a cancellation-free ``log(1 - exp(x))``. Far enough out that bracket
    tends to $1/z^2$, and the expression reduces to its asymptote.

    Parameters
    ----------
    z : np.ndarray
        Standardised improvement.

    Returns
    -------
    log_h : np.ndarray
    """
    z = np.asarray(z, dtype=float)
    result = np.empty_like(z)

    # Close to and above zero the direct form is accurate and nothing underflows.
    direct = z > -1.0
    if np.any(direct):
        z_direct = z[direct]
        result[direct] = np.log(norm.pdf(z_direct) + z_direct * norm.cdf(z_direct))

    stable = (~direct) & (z >= _ASYMPTOTIC_THRESHOLD)
    if np.any(stable):
        a = -z[stable]
        inner = np.log(a) + _LOG_SQRT_PI_2 + np.log(erfcx(a / math.sqrt(2.0)))
        result[stable] = -0.5 * a**2 - _LOG_SQRT_2PI + _log1mexp(inner)

    asymptotic = z < _ASYMPTOTIC_THRESHOLD
    if np.any(asymptotic):
        a = -z[asymptotic]
        result[asymptotic] = -0.5 * a**2 - _LOG_SQRT_2PI - 2.0 * np.log(a)

    return result


class LogEI(AbstractAcquisitionFunction):
    r"""Logarithm of the expected improvement.

    $$
    \log EI(\mathbf{X}) = \log \sigma(\mathbf{X})
                          + \log h\!\left(\frac{\eta - \mu(\mathbf{X})}{\sigma(\mathbf{X})}\right)
    $$

    Ranks configurations exactly as :class:`EI` does, since the logarithm is increasing, but keeps discriminating
    in the regions where expected improvement underflows to zero and leaves the maximizer with a flat surface
    to search. See "Unexpected Improvements to Expected Improvement for Bayesian Optimization" by Sebastian
    Ament et al. [[ADE+23][ADE+23]].

    Note
    ----
    This is not the same thing as ``EI(log=True)``, which computes ordinary expected improvement for a model
    whose *target values* have been log scaled, and returns a plain, non-negative acquisition value. This class
    returns the logarithm of the acquisition value itself, and so returns negative numbers.

    Parameters
    ----------
    xi : float, defaults to 0.0
        Controls the balance between exploration and exploitation.
    """

    def __init__(self, xi: float = 0.0) -> None:
        super().__init__()

        self._xi = xi
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return "Log Expected Improvement"

    @property
    def log(self) -> bool:  # noqa: D102
        return True

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"xi": self._xi})

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        eta : float
            Function value of the current incumbent.
        xi : float, optional
            Exploration-exploitation trade-off parameter.
        """
        assert "eta" in kwargs
        self._eta = kwargs["eta"]

        if "xi" in kwargs and kwargs["xi"] is not None:
            self._xi = kwargs["xi"]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the log expected improvement.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated.

        Returns
        -------
        np.ndarray [N, 1]
            Log expected improvement of X. Negative, unlike every non-logarithmic acquisition function.

        Raises
        ------
        ValueError
            If `update` has not been called before (current incumbent value `eta` unspecified).
        """
        assert self._model is not None

        if self._eta is None:
            raise ValueError(
                "No current best specified. Call update("
                "eta=<int>) to inform the acquisition function "
                "about the current best value."
            )

        if len(X.shape) == 1:
            X = X[:, np.newaxis]

        means, variances = self._model.predict_marginalized(X)
        stds = np.sqrt(variances)

        # A zero standard deviation means the point has been observed everywhere, so there is nothing left to
        # gain there. Log of zero improvement is negative infinity, which ranks it below any real candidate.
        degenerate = stds <= 0.0
        safe_stds = np.where(degenerate, 1.0, stds)

        z = (self._eta - means - self._xi) / safe_stds
        values = np.log(safe_stds) + log_h(z)

        return np.where(degenerate, -np.inf, values).reshape((-1, 1))
