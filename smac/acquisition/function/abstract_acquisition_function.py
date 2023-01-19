from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from ConfigSpace import Configuration

from smac.model.abstract_model import AbstractModel
from smac.utils.configspace import convert_configurations_to_array
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AbstractAcquisitionFunction:
    """Abstract base class for acquisition function."""

    def __init__(self) -> None:
        self._model: AbstractModel | None = None

    @property
    def name(self) -> str:
        """Returns the full name of the acquisition function."""
        raise NotImplementedError

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
        attributes of the acqusition function which are already present.

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
        """Update acsquisition function attributes

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
            Acquisition function values wrt X.
        """
        raise NotImplementedError
