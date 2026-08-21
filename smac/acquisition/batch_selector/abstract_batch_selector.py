from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AbstractBatchSelector:
    """Abstract class for selecting a batch of configurations jointly.

    The acquisition function scores every candidate on its own, so sorting by acquisition value
    and taking the first ``batch_size`` entries tends to return neighbours on the same mode. A
    batch selector replaces that final ordering step with one that accounts for the batch as a
    whole, either by considering the joint posterior over the candidates or by randomising the
    order in a controlled way.

    Note
    ----
    ``select`` returns *all* candidates in a new order rather than only the chosen batch. The
    first ``batch_size`` entries are the batch; the remainder is a tail that lets the caller keep
    drawing configurations when some of the batch has been evaluated before. This keeps the
    existing retry machinery in ``ConfigSelector`` working as a backstop.

    Parameters
    ----------
    seed : int, defaults to 0
        Random seed.
    """

    def __init__(self, seed: int = 0) -> None:
        self._seed = seed
        self._rng = np.random.RandomState(seed=seed)

    @property
    def name(self) -> str:
        """Returns the full name of the batch selector."""
        raise NotImplementedError

    @property
    def requires_joint_samples(self) -> bool:
        """Whether this selector needs a model that supports joint posterior samples."""
        return False

    @property
    def requires_conditioning(self) -> bool:
        """Whether this selector needs a model that supports conditioning."""
        return False

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
            "seed": self._seed,
        }

    def select(
        self,
        candidates: list[Configuration],
        acquisition_values: np.ndarray,
        batch_size: int,
        model: AbstractModel | None = None,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        pending: list[Configuration] | None = None,
    ) -> list[Configuration]:
        """Reorders the candidates so that the first ``batch_size`` entries form the batch.

        Calls ``_select``, implemented by a subclass, after validating that the model provides
        whatever the selector needs.

        Parameters
        ----------
        candidates : list[Configuration]
            Candidate configurations, ordered by descending acquisition value.
        acquisition_values : np.ndarray [#candidates, ]
            The acquisition value of each candidate, in the same order.
        batch_size : int
            How many configurations are wanted at once.
        model : AbstractModel | None, defaults to None
            The surrogate model the acquisition values came from.
        acquisition_function : AbstractAcquisitionFunction | None, defaults to None
            The acquisition function the values came from. Needed by selectors that re-score
            candidates against a changed model.
        pending : list[Configuration] | None, defaults to None
            Configurations that are currently being evaluated. How these are taken into account
            differs per selector and is documented on each of them.

        Returns
        -------
        candidates : list[Configuration]
            All candidates, reordered.
        """
        pending = pending if pending is not None else []
        acquisition_values = np.asarray(acquisition_values, dtype=float).flatten()

        if len(candidates) != len(acquisition_values):
            raise ValueError(
                f"Got {len(candidates)} candidates but {len(acquisition_values)} acquisition values."
            )

        if len(candidates) <= 1 or batch_size <= 1:
            return list(candidates)

        if self.requires_joint_samples and (model is None or not model.supports_joint_samples):
            raise ValueError(
                f"{self.__class__.__name__} needs a model whose `supports_joint_samples` is True. "
                f"Got {None if model is None else model.__class__.__name__}."
            )

        if self.requires_conditioning and (model is None or not model.supports_conditioning):
            raise ValueError(
                f"{self.__class__.__name__} needs a model whose `supports_conditioning` is True. "
                f"Got {None if model is None else model.__class__.__name__}. Selectors based on "
                "pseudo-observations produce identical batch members on models that cannot be "
                "conditioned in closed form."
            )

        return self._select(
            candidates=candidates,
            acquisition_values=acquisition_values,
            batch_size=batch_size,
            model=model,
            acquisition_function=acquisition_function,
            pending=pending,
        )

    @abstractmethod
    def _select(
        self,
        candidates: list[Configuration],
        acquisition_values: np.ndarray,
        batch_size: int,
        model: AbstractModel | None,
        acquisition_function: AbstractAcquisitionFunction | None,
        pending: list[Configuration],
    ) -> list[Configuration]:
        """Implements the reordering. See ``select`` for the parameters."""
        raise NotImplementedError()

    def _order_by(self, candidates: list[Configuration], scores: np.ndarray) -> list[Configuration]:
        """Returns the candidates ordered by descending score, breaking ties randomly."""
        tie_break = self._rng.rand(len(scores))
        indices = np.lexsort((tie_break, scores))

        return [candidates[i] for i in indices[::-1]]
