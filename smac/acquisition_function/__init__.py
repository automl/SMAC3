from __future__ import annotations

import abc
from typing import Any, List, Tuple

import numpy as np

from smac.configspace import Configuration
from smac.configspace.util import convert_configurations_to_array
from smac.model.base_model import BaseModel
from smac.utils.logging import PickableLoggerAdapter

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class AbstractAcquisitionFunction(metaclass=abc.ABCMeta):
    """Abstract base class for acquisition function.

    Parameters
    ----------
    model : BaseEPM
        Models the objective function.

    Attributes
    ----------
    model
    logger
    """

    def __init__(self) -> None:
        self.model: BaseModel | None = None
        self._required_updates = ("model",)  # type: Tuple[str, ...]
        self.logger = PickableLoggerAdapter(self.__module__ + "." + self.__class__.__name__)

    def set_model(self, model) -> None:
        self.model = model

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        This method will be called after fitting the model, but before maximizing the acquisition
        function. As an examples, EI uses it to update the current fmin.

        The default implementation only updates the attributes of the acqusition function which
        are already present.

        Parameters
        ----------
        kwargs
        """
        for key in self._required_updates:
            if key not in kwargs:
                raise ValueError(
                    "Acquisition function %s needs to be updated with key %s, but only got "
                    "keys %s." % (self.__class__.__name__, key, list(kwargs.keys()))
                )
        for key in kwargs:
            if key in self._required_updates:
                setattr(self, key, kwargs[key])

    def __call__(self, configurations: List[Configuration]) -> np.ndarray:
        """Computes the acquisition value for a given X.

        Parameters
        ----------
        configurations : list
            The configurations where the acquisition function
            should be evaluated.

        Returns
        -------
        np.ndarray(N, 1)
            acquisition values for X
        """
        X = convert_configurations_to_array(configurations)
        if len(X.shape) == 1:
            X = X[np.newaxis, :]

        acq = self._compute(X)
        if np.any(np.isnan(acq)):
            idx = np.where(np.isnan(acq))[0]
            acq[idx, :] = -np.finfo(float).max
        return acq

    @abc.abstractmethod
    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the acquisition value for a given point X. This function has to be overwritten
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
        raise NotImplementedError()
