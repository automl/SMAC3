from __future__ import annotations

from abc import abstractmethod, ABCMeta
from typing import Any

import numpy as np

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"


logger = get_logger(__name__)


class AbstractAcquisitionFunction(metaclass=ABCMeta):
    """Abstract base class for acquisition function.

    Parameters
    ----------
    model : BaseModel
        Models the objective function.

    Attributes
    ----------
    model : BaseModel
    """

    def __init__(self) -> None:
        self.model: AbstractModel | None = None

    def _set_model(self, model: AbstractModel) -> None:
        self.model = model

    @abstractmethod
    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        raise NotImplementedError

    @abstractmethod
    def update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function. As an examples, EI uses it to update the current fmin.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        Parameters
        ----------
        kwargs : Any
        """
        raise NotImplementedError

    def __call__(self, configurations: list[Configuration]) -> np.ndarray:
        """Compute the acquisition value for a given X.

        Parameters
        ----------
        configurations : list[Configuration]
            The configurations where the acquisition function
            should be evaluated.

        Returns
        -------
        np.ndarray(N, 1)
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
        X : np.ndarray
            The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Acquisition function values wrt X
        """
        raise NotImplementedError
