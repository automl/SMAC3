from __future__ import annotations

from typing import Any, Iterable, Literal, Mapping, Sequence

import numpy as np

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
        active = self._active_members()

        if len(active) == 0:
            return np.ones((X.shape[0], 1))

        # Each member floors and decays itself before the combination: that is what lets members which arrived at
        # different times carry different amounts of influence.
        values = np.concatenate([weight(X).reshape((-1, 1)) for weight in active], axis=1)

        if self._combination == "sum":
            combined = np.sum(values, axis=1)
        elif self._combination == "product":
            combined = np.prod(values, axis=1)
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
