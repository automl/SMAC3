from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np
from scipy.special import logsumexp

from smac.acquisition.weight.abstract_weight import AbstractAcquisitionWeight
from smac.acquisition.weight.decay import DecaySchedule
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)

Combination = Literal["sum", "product", "max"]

_COMBINATIONS: tuple[str, ...] = ("sum", "product", "max")


class CompositeWeight(AbstractAcquisitionWeight):
    r"""Several weights of the same kind, combined among themselves before entering the acquisition function.

    `WeightedAcquisitionFunction` always multiplies its weights together, because that is what a multiplicative
    weight means. Weights which should combine some other way say so by living inside a composite: the members are
    combined by `combination`, and the composite as a whole is then multiplied in like any other weight. The rule
    is therefore visible in the object graph rather than hidden behind a flag, and a rule which only makes sense
    among weights of one kind cannot accidentally be applied across kinds.

    Every member applies its own floor and its own decay from its own anchor *before* the combination, which is
    what lets members that arrived at different times carry different amounts of influence.

    Members are keyed, so that one can be replaced or removed while the optimization is running.

    Parameters
    ----------
    weights : Sequence[AbstractAcquisitionWeight] | Mapping[str, AbstractAcquisitionWeight] | None, defaults to None
        The initial members. A sequence is keyed automatically.
    combination : {"sum", "product", "max"}, defaults to "sum"
        How the members are combined. Summing reads as "any of these is interesting", multiplying as "all of these
        have to hold", and taking the maximum as "the most enthusiastic member decides".
    floor : float, defaults to 0.0
        Lowest possible value of the combination. Members carry their own floors, so this is zero by default.
    decay : DecaySchedule | None, defaults to None
        Decay applied to the combination on top of each member's own decay. Rarely useful.
    """

    def __init__(
        self,
        weights: Sequence[AbstractAcquisitionWeight] | Mapping[str, AbstractAcquisitionWeight] | None = None,
        *,
        combination: Combination = "sum",
        floor: float = 0.0,
        decay: DecaySchedule | None = None,
    ) -> None:
        super().__init__(floor=floor, decay=decay)

        if combination not in _COMBINATIONS:
            raise ValueError(f"Unknown combination {combination!r}. Available combinations are {_COMBINATIONS}.")

        self._combination: Combination = combination
        self._members: dict[str, AbstractAcquisitionWeight] = {}
        self._counter = 0

        # The last arguments the composite was updated with, replayed onto a member added later so that it is
        # immediately usable instead of waiting for the next iteration.
        self._last_update: dict[str, Any] | None = None

        if weights is not None:
            items: Iterable[tuple[str | None, AbstractAcquisitionWeight]]
            if isinstance(weights, Mapping):
                items = [(key, weight) for key, weight in weights.items()]
            else:
                items = [(None, weight) for weight in weights]

            for key, weight in items:
                self.add(weight, key=key)

    @property
    def name(self) -> str:  # noqa: D102
        return f"{self.__class__.__name__} ({self._combination} of {len(self._members)})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "combination": self._combination,
                "weights": {key: weight.meta for key, weight in self._members.items()},
            }
        )

        return meta

    @property
    def members(self) -> Mapping[str, AbstractAcquisitionWeight]:
        """The members, by key, in insertion order."""
        return dict(self._members)

    @property
    def combination(self) -> Combination:
        """How the members are combined."""
        return self._combination

    def __len__(self) -> int:
        return len(self._members)

    def add(self, weight: AbstractAcquisitionWeight, *, key: str | None = None) -> str:
        """Adds a member.

        The member is brought up to date immediately, using the arguments of the last update, so that a weight added
        between two iterations is usable straight away.

        Parameters
        ----------
        weight : AbstractAcquisitionWeight
            The member to add.
        key : str | None, defaults to None
            Key to register it under. Generated if not given.

        Returns
        -------
        str
            The key the member is registered under.
        """
        if key is None:
            self._counter += 1
            key = f"{weight.name}-{self._counter}"

        if key in self._members:
            raise ValueError(f"A weight is already registered under the key {key!r}.")

        self._members[key] = weight

        if self._last_update is not None and self._model is not None:
            weight.update(model=self._model, **self._last_update)

        return key

    def remove(self, key: str) -> AbstractAcquisitionWeight:
        """Removes the member registered under the given key and returns it."""
        if key not in self._members:
            raise KeyError(f"No weight is registered under the key {key!r}.")

        return self._members.pop(key)

    def replace(self, key: str, weight: AbstractAcquisitionWeight) -> None:
        """Replaces the member registered under the given key, keeping its position."""
        if key not in self._members:
            raise KeyError(f"No weight is registered under the key {key!r}.")

        self._members[key] = weight

        if self._last_update is not None and self._model is not None:
            weight.update(model=self._model, **self._last_update)

    def clear(self) -> None:
        """Removes every member."""
        self._members.clear()

    @property
    def model(self) -> AbstractModel | None:  # noqa: D102
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        self._model = model

        for weight in tuple(self._members.values()):
            weight.model = model

    def update(self, model: AbstractModel, **kwargs: Any) -> None:  # noqa: D102
        self._last_update = dict(kwargs)
        super().update(model, **kwargs)

    def _update(self, **kwargs: Any) -> None:
        assert self._model is not None

        for weight in tuple(self._members.values()):
            weight.update(model=self._model, **kwargs)

    def _active_members(self) -> list[AbstractAcquisitionWeight]:
        return [weight for weight in tuple(self._members.values()) if weight.is_active()]

    def _compute(self, X: np.ndarray) -> np.ndarray:
        return self._combine(X, log=False)

    def _compute_log(self, X: np.ndarray) -> np.ndarray:
        """The combination performed in log space, rather than its logarithm taken afterwards.

        Taking `log(_compute(X))` would form the very products and sums the logarithm exists to avoid, so the
        rule itself is translated instead. Each rule has exactly one image under the logarithm:

        ============  ==================
        combination   in log space
        ============  ==================
        ``sum``       ``logsumexp``
        ``product``   ``sum``
        ``max``       ``max``
        ============  ==================

        The middle row is the one worth pausing on, and the reason this is a translated rule rather than a
        shared one: a *product* of weights becomes a *sum* of logarithms, so a composite that multiplies and
        one that adds swap places here. Reusing `_compute`'s branch names in log space would silently give
        every ensemble the wrong rule.
        """
        return self._combine(X, log=True)

    def _combine(self, X: np.ndarray, *, log: bool) -> np.ndarray:
        active = self._active_members()

        # One everywhere, whose logarithm is zero: a composite with nothing to say must leave whatever it is
        # combined with exactly as it was, in either space.
        if len(active) == 0:
            return np.zeros((X.shape[0], 1)) if log else np.ones((X.shape[0], 1))

        # Each member floors and decays itself before the combination: that is what lets members which arrived at
        # different times carry different amounts of influence.
        values = np.concatenate([weight(X, log=log).reshape((-1, 1)) for weight in active], axis=1)

        if self._combination == "sum":
            combined = logsumexp(values, axis=1) if log else np.sum(values, axis=1)
        elif self._combination == "product":
            combined = np.sum(values, axis=1) if log else np.prod(values, axis=1)
        else:
            combined = np.max(values, axis=1)

        return combined.reshape((-1, 1))

    def is_active(self) -> bool:
        """Whether any member has anything to say."""
        return len(self._active_members()) > 0

    def suppresses_acquisition(self) -> bool:
        """Whether any member wants to take over the ranking."""
        return any(weight.suppresses_acquisition() for weight in self._active_members())

    def adjust_eta(self, eta: float | None) -> float | None:
        """Lets every member correct the incumbent value in turn."""
        for weight in self._active_members():
            eta = weight.adjust_eta(eta)

        return eta


class PriorEnsemble(CompositeWeight):
    r"""Several user beliefs about where the optimum lies, each fading from when it was supplied.

    $$w(\mathbf{X}) = \sum_m \pi_m(\mathbf{X})^{\beta_m / (t - t_{0,m} + 1)}$$

    See "Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization" by Lukas Fehring et al.
    [[FWS+25][FWS+25]] for further details.

    Beliefs are summed rather than multiplied, so that the ensemble reads as "any of these regions is worth a
    look". Multiplying would let two beliefs pointing at different regions cancel each other out, which is the
    wrong answer when a user names a second promising region without retracting the first.

    Be aware of what summing implies as beliefs age: a fully decayed belief contributes about one everywhere,
    while a freshly supplied sharp belief contributes far less than one almost everywhere, so a pile of stale
    beliefs can drown out the newest and most informative one. `prune_exponent` drops beliefs whose exponent has
    fallen below a threshold, and `combination="max"` lets the most enthusiastic belief decide instead.

    Parameters
    ----------
    priors : Sequence[AbstractAcquisitionWeight] | Mapping[str, AbstractAcquisitionWeight] | None, defaults to None
        The initial beliefs.
    combination : {"sum", "product", "max"}, defaults to "sum"
        How the beliefs are combined.
    prune_exponent : float | None, defaults to None
        Drop a belief once its decay exponent falls below this. `None` keeps every belief forever, which is what
        the formula above says.
    """

    def __init__(
        self,
        priors: Sequence[AbstractAcquisitionWeight] | Mapping[str, AbstractAcquisitionWeight] | None = None,
        *,
        combination: Combination = "sum",
        prune_exponent: float | None = None,
    ) -> None:
        self._prune_exponent = prune_exponent

        super().__init__(priors, combination=combination)

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"prune_exponent": self._prune_exponent})

        return meta

    def _update(self, **kwargs: Any) -> None:
        super()._update(**kwargs)

        if self._prune_exponent is None:
            return

        for key, weight in list(self._members.items()):
            if weight.decay(weight.steps) < self._prune_exponent:
                logger.debug(f"Dropping the prior {key!r}, whose influence has decayed away.")
                self.remove(key)
