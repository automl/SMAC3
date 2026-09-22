from __future__ import annotations

from abc import abstractmethod
from enum import Enum
from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.model.abstract_model import AbstractModel
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AcquisitionScale(str, Enum):
    r"""What an acquisition function's values mean, and so how a weight combines with them.

    A weight - the probability that an output constraint holds, a user's belief about where the optimum is -
    is a non-negative factor that should make a configuration look better or worse. Which arithmetic achieves
    that depends entirely on what the acquisition values are, and getting it wrong is silent: the search keeps
    running and steers the wrong way.

    Three conventions are in use, and they are exhaustive and mutually exclusive, which is why this is one
    property rather than a set of flags. Two booleans would admit "logarithmic *and* signed", which describes
    nothing, and would leave every wrapper to decide what that meant.

    ``LINEAR``
        Non-negative, proportional to how good a configuration looks. Expected improvement, probability of
        improvement. A weight **multiplies**: $a(x) \cdot w(x)$.

    ``LOG``
        Logarithms of the above, so negative, and ordered identically because the logarithm is increasing.
        `LogEI` [[ADE+23][ADE+23]]. A weight **adds its own logarithm**: $\log a(x) + \log w(x)$. This is the
        convention that survives a sharp weight - a belief narrow enough drives $a(x) \cdot w(x)$ to exactly
        zero across most of the space in floating point, leaving the maximizer a flat surface to climb, while
        the sum stays finite and ordered.

    ``SIGNED``
        Negative by construction, because the quantity described is a cost and the maximizer maximizes.
        Confidence bounds, Thompson sampling. Multiplying these by a weight in $[0, 1]$ moves them *up*, so a
        weight would favour exactly what it is meant to discourage. The values are **shifted by the incumbent
        first** and then multiplied, which puts them back on the linear convention.
    """

    LINEAR = "linear"
    LOG = "log"
    SIGNED = "signed"


class AbstractAcquisitionFunction:
    """Abstract base class for acquisition function."""

    def __init__(self) -> None:
        self._model: AbstractModel | None = None

    @property
    def name(self) -> str:
        """Returns the full name of the acquisition function."""
        raise NotImplementedError

    @property
    def value_scale(self) -> AcquisitionScale:
        """What the returned values mean, and therefore how a weight combines with them.

        Everything that wraps an acquisition function to reweight it - an output constraint, a user prior -
        has to know this before it can touch a value, and the three answers call for three different
        arithmetic. See `AcquisitionScale`.
        """
        return AcquisitionScale.LINEAR

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    @property
    def model(self) -> AbstractModel | None:
        """Return the used surrogate model in the acquisition function."""
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        """Updates the surrogate model."""
        self._model = model

    def update(self, model: AbstractModel, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function. As an examples, EI uses it to update the current fmin. The default implementation only updates the
        attributes of the acquisition function which are already present.

        Calls `_update` to update the acquisition function attributes.

        Parameters
        ----------
        model : AbstractModel
            The model which was used to fit the data.
        kwargs : Any
            Additional arguments to update the specific acquisition function.
        """
        self.model = model
        self._update(**kwargs)

    def _update(self, **kwargs: Any) -> None:
        """Update acquisition function attributes

        Might be different for each child class.
        """
        pass

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        """Compute the acquisition value for a given configuration.

        Parameters
        ----------
        configurations : list[Configuration]
            The configurations where the acquisition function should be evaluated.

        Returns
        -------
        np.ndarray [N, 1]
            Acquisition values for X
        """
        X = convert_configurations_to_array(configurations)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max

        return acq

    @abstractmethod
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the acquisition value for a given point X. This function has to be overwritten
        in a derived class.

        Parameters
        ----------
        X : np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N,1]
            Acquisition function values wrt X. Larger is better. What the numbers themselves mean - and so how
            anything weighting them has to combine with them - is declared by ``value_scale``; either way only
            their order matters to the acquisition maximizer.
        """
        raise NotImplementedError
