from __future__ import annotations

from abc import abstractmethod
from typing import Any, Callable

import numpy as np

from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class DecaySchedule:
    r"""Exponent applied to a multiplicative acquisition weight over time.

    A weight $w(\mathbf{X})$ enters the acquisition function as $w(\mathbf{X})^{e(k)}$, where $k$ is the number of
    trials since the weight was anchored. Driving the weight to one rather than removing it keeps the acquisition
    function continuous, which is what preserves the convergence guarantees of the unweighted optimization.

    A schedule never knows *when* its weight was anchored; it only sees the number of steps since. The weight owns
    the anchor. That split is what lets a prior supplied before the first trial and a prior supplied halfway through
    a run share the same schedule objects.
    """

    @abstractmethod
    def exponent(self, steps: int) -> float:
        """Returns the exponent to raise the weight to.

        Parameters
        ----------
        steps : int
            Number of trials since the weight was anchored. Never negative.

        Returns
        -------
        float
            The exponent. One leaves the weight unchanged.
        """
        raise NotImplementedError

    def __call__(self, steps: int) -> float:
        """Returns the exponent for a possibly negative number of steps, clamped at zero."""
        return self.exponent(max(0, steps))

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"name": self.__class__.__name__}


class NoDecay(DecaySchedule):
    """Leaves the weight untouched for the whole run.

    Used by weights which express a fact about the problem rather than a belief about it, such as the probability
    that an output constraint holds: a constraint does not become less true as the run progresses.
    """

    def exponent(self, steps: int) -> float:  # noqa: D102
        return 1.0


class PolynomialDecay(DecaySchedule):
    r"""Decays the weight as $\beta / (k + \text{offset})^{\text{power}}$.

    With the defaults this is the piBO schedule [[HSSL22][HSSL22]], $\beta / (k + 1)$. Higher powers are the
    faster variants evaluated by DynaBO [[FWS+25][FWS+25]], which hand control back to the surrogate sooner.

    Parameters
    ----------
    beta : float
        Decay factor. A solid default (empirically founded) is ``scenario.n_trials`` / 10. Larger values keep the
        weight influential for longer.
    power : float, defaults to 1.0
        Power of the denominator. One is linear, two quadratic, and so on.
    offset : float, defaults to 1.0
        Added to the number of steps, so that the exponent is finite at the anchor itself.
    """

    def __init__(self, beta: float, power: float = 1.0, offset: float = 1.0) -> None:
        if beta <= 0:
            raise ValueError(f"The decay factor must be positive, got {beta}.")

        if power <= 0:
            raise ValueError(f"The decay power must be positive, got {power}.")

        if offset <= 0:
            raise ValueError(f"The decay offset must be positive, got {offset}.")

        self._beta = beta
        self._power = power
        self._offset = offset

    def exponent(self, steps: int) -> float:  # noqa: D102
        return float(self._beta / np.power(steps + self._offset, self._power))

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"beta": self._beta, "power": self._power, "offset": self._offset})

        return meta


class LogarithmicDecay(DecaySchedule):
    r"""Decays the weight as $\beta / \log(k + \text{offset} + 1)$.

    The slowest of the schedules evaluated by DynaBO [[FWS+25][FWS+25]]: the weight keeps a noticeable influence
    long after a polynomial schedule has flattened it.

    Parameters
    ----------
    beta : float
        Decay factor. Larger values keep the weight influential for longer.
    offset : float, defaults to 1.0
        Added to the number of steps, so that the logarithm stays positive at the anchor itself.
    """

    def __init__(self, beta: float, offset: float = 1.0) -> None:
        if beta <= 0:
            raise ValueError(f"The decay factor must be positive, got {beta}.")

        if offset <= 0:
            raise ValueError(f"The decay offset must be positive, got {offset}.")

        self._beta = beta
        self._offset = offset

    def exponent(self, steps: int) -> float:  # noqa: D102
        return float(self._beta / np.log(steps + self._offset + 1))

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"beta": self._beta, "offset": self._offset})

        return meta


DECAY_SHAPES: dict[str, Callable[[float], DecaySchedule]] = {
    "logarithmic": lambda beta: LogarithmicDecay(beta),
    "linear": lambda beta: PolynomialDecay(beta, power=1.0),
    "quadratic": lambda beta: PolynomialDecay(beta, power=2.0),
    "cubic": lambda beta: PolynomialDecay(beta, power=3.0),
    "quartic": lambda beta: PolynomialDecay(beta, power=4.0),
    "quintic": lambda beta: PolynomialDecay(beta, power=5.0),
}
"""The decay shapes evaluated by DynaBO, by name, so that a schedule can be selected from a configuration file or
a user interface without importing the classes."""


def get_decay_schedule(shape: str, beta: float) -> DecaySchedule:
    """Returns the named decay schedule with the given decay factor.

    Parameters
    ----------
    shape : str
        One of the keys of ``DECAY_SHAPES``.
    beta : float
        Decay factor to construct the schedule with.

    Returns
    -------
    DecaySchedule
    """
    if shape not in DECAY_SHAPES:
        raise ValueError(f"Unknown decay shape {shape!r}. Available shapes are {sorted(DECAY_SHAPES)}.")

    return DECAY_SHAPES[shape](beta)
