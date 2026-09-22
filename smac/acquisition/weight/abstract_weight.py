from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np

from smac.acquisition.weight.decay import DecaySchedule, NoDecay
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractAcquisitionWeight:
    r"""A non-negative factor multiplied into an acquisition function.

    Several things SMAC does to steer the search have this shape. An output constraint contributes the probability
    that its bound holds; a user prior over the optimum contributes a density. Both leave the surrogate model of the
    objective untouched and only reweight how interesting each configuration looks, which is why they can be
    expressed as one kind of object and applied together by `WeightedAcquisitionFunction`.

    A weight is evaluated as

    $$w(\mathbf{X}) = \left( \tilde{w}(\mathbf{X}) + \text{floor} \right)^{e(k)}$$

    where $\tilde{w}$ is `_compute`, and $e(k)$ is the decay exponent after $k$ steps. The floor is added *before*
    the exponent is applied: zero raised to any positive power is still zero, so flooring afterwards would leave the
    weight exactly zero and lose the ranking the floor exists to preserve.

    Beyond producing a value, a weight negotiates three things with the acquisition function it is applied to.
    `is_active` lets it stand aside entirely while it has nothing to say, `suppresses_acquisition` lets it take over
    the ranking when the acquisition function itself is meaningless, and `adjust_eta` lets it correct the incumbent
    value the acquisition function improves over.

    Parameters
    ----------
    floor : float, defaults to 1e-12
        Lowest possible value of the weight. Keeps the ranking of configurations intact when the weight underflows
        to zero everywhere.
    decay : DecaySchedule | None, defaults to None
        How the weight fades as trials accumulate. `None` means `NoDecay`, for a weight which states a fact about
        the problem rather than a belief about it.
    """

    def __init__(self, *, floor: float = 1e-12, decay: DecaySchedule | None = None) -> None:
        if floor < 0:
            raise ValueError(f"The floor must not be negative, got {floor}.")

        self._model: AbstractModel | None = None
        self._floor = floor
        self._decay: DecaySchedule = decay if decay is not None else NoDecay()
        self._steps = 0

    @property
    def name(self) -> str:
        """Returns the full name of the weight."""
        return self.__class__.__name__

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "floor": self._floor,
            "decay": self._decay.meta,
        }

    @property
    def model(self) -> AbstractModel | None:
        """The surrogate model of the objective.

        Note this is *not* a model of whatever the weight itself describes; a weight which needs one of those owns it
        separately.
        """
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        """Updates the surrogate model of the objective."""
        self._model = model

    @property
    def decay(self) -> DecaySchedule:
        """The decay schedule of the weight."""
        return self._decay

    @property
    def log_floor(self) -> float:
        """The floor to apply in log space, as a logarithm. `-inf` means none.

        Separate from `floor` because the floor does not mean the same thing in both spaces, and a single
        number cannot serve both. In linear space it stops a weight collapsing to exactly zero, which would
        make every point below it tie regardless of how the acquisition function ranked them. In log space
        nothing collapses, so a floor there only discards information - which is right for a belief, whose
        floor exists so a mistaken user can be recovered from, and wrong for a fact like feasibility, which
        should be free to say that a region is hopeless by many orders of magnitude.

        The default mirrors `floor`, which is the conservative reading; a weight that wants no floor in log
        space overrides this and says why.
        """
        return -np.inf if self._floor <= 0.0 else float(np.log(self._floor))

    @property
    def steps(self) -> int:
        """Number of trials since the weight was anchored, as of the last update."""
        return self._steps

    def update(self, model: AbstractModel, **kwargs: Any) -> None:
        """Prepares the weight for the next acquisition function maximization.

        Called once per iteration, before the wrapped acquisition function is updated, with the keyword arguments
        `ConfigSelector` already passes: ``eta``, ``num_data``, ``X``, ``incumbents``, ``incumbent_array``,
        ``runhistory`` and ``runhistory_encoder``.

        Parameters
        ----------
        model : AbstractModel
            The surrogate model of the objective, which was just fitted to the data.
        """
        self.model = model
        self._steps = self._compute_steps(**kwargs)
        self._update(**kwargs)

    def _update(self, **kwargs: Any) -> None:
        """Updates the attributes of the weight. Might be different for each child class."""
        pass

    def _compute_steps(self, **kwargs: Any) -> int:
        """Returns the number of trials since the weight was anchored. Zero for a weight which does not decay."""
        return 0

    def __call__(self, X: np.ndarray, *, log: bool = False) -> np.ndarray:
        """Computes the floored and decayed weight, or its logarithm.

        One method rather than two, because the two differ only in which arithmetic expresses the same
        quantity: the decay raises the weight to a power, and the logarithm of that is the power times the
        logarithm. A second method would be the same decision written twice, and a weight that overrode one
        and not the other would be silently inconsistent.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The points to evaluate the weight at.
        log : bool, defaults to False
            Return $\\log w(\\mathbf{X})$ instead of $w(\\mathbf{X})$, for an acquisition function whose own
            values are logarithms. This is not a convenience: a sharp weight underflows to exactly zero in
            the linear form long before its logarithm stops being representable, and a weight that is zero
            everywhere ranks nothing.

        Returns
        -------
        np.ndarray [N, 1]
            The weight of X, non-negative; or its logarithm, which is not.
        """
        exponent = self._decay(self._steps)

        if log:
            # Floored in log space rather than by flooring the value and taking its logarithm, because the
            # value is exactly what may have underflowed: log(0 + floor) would report the floor as though it
            # were the answer, discarding a perfectly representable -5000 to report -27.6.
            raw = np.asarray(self._compute_log(X), dtype=float).reshape((-1, 1))
            floor = self.log_floor
            floored = raw if floor == -np.inf else np.maximum(raw, floor)

            return floored if exponent == 1.0 else exponent * floored

        raw = np.asarray(self._compute(X), dtype=float).reshape((-1, 1))

        if exponent == 1.0:
            return raw + self._floor

        return np.power(raw + self._floor, exponent)

    def _compute_log(self, X: np.ndarray) -> np.ndarray:
        """Computes the logarithm of the raw weight, before the floor and the decay are applied.

        The default takes the logarithm of `_compute`, which is correct but only as good as the value it is
        given: a weight that has already underflowed to zero reports `-inf` here, having lost the very
        distinctions the logarithm exists to preserve. A weight whose logarithm has a stable closed form -
        a Gaussian probability through `log_ndtr`, say - should override this and compute it directly. That
        is the whole benefit, so the override is the point rather than an optimization.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The points to evaluate the weight at.

        Returns
        -------
        np.ndarray [N, 1]
            Logarithm of the raw weight of X.
        """
        with np.errstate(divide="ignore"):
            return np.log(np.asarray(self._compute(X), dtype=float).reshape((-1, 1)))

    @abstractmethod
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the raw weight, before the floor and the decay are applied.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The points to evaluate the weight at.

        Returns
        -------
        np.ndarray [N, 1]
            Non-negative raw weight of X.
        """
        raise NotImplementedError

    def is_active(self) -> bool:
        """Whether the weight has anything to say this iteration.

        An inactive weight is skipped entirely, exactly as if it had not been added, rather than contributing a
        constant one. The difference matters: a constant weight would still force the acquisition values to be
        rescaled, which clips them at zero.
        """
        return True

    def suppresses_acquisition(self) -> bool:
        """Whether the acquisition value should be replaced by one, leaving the weights to rank the candidates.

        For the case where the acquisition function has nothing to say - there is no improvement to expect over an
        incumbent which does not exist yet - and the search should be driven by the weights alone.
        """
        return False

    def adjust_eta(self, eta: float | None) -> float | None:
        """Returns the incumbent value to hand to the wrapped acquisition function.

        A weight which knows that some observed configurations do not count may correct the incumbent here, so that
        the acquisition function does not aim at a target it should not.
        """
        return eta
