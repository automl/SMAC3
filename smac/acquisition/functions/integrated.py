from __future__ import annotations

from typing import Any
import copy
import numpy as np

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class IntegratedAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Marginalize over Model hyperparameters to compute the integrated acquisition function.

    See "Practical Bayesian Optimization of Machine Learning Algorithms" by Jasper Snoek et al.
    (https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
    for further details.

    Parameters
    ----------
    long_name : str
    acquisition_function : AbstractAcquisitionFunction
    eta : float
        Current incumbent value.
    """

    def __init__(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        super().__init__()
        self.long_name : str = f"Integrated Acquisition Function ({acquisition_function.__class__.__name__})"
        self.acquisition_function : AbstractAcquisitionFunction = acquisition_function
        self.eta: float | None = None
        self._functions : list[AbstractAcquisitionFunction] = [] 

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def update(self, **kwargs: Any) -> None:
        """Update the acquisition functions values.

        This method will be called if the model is updated. E.g. entropy search uses it to update its approximation
        of P(x=x_min), EI uses it to update the current fmin.

        This implementation creates an acquisition function object for each model to integrate over and sets the
        respective attributes for each acquisition function object.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for 
        """
        model = kwargs["model"]
        del kwargs["model"]
        if not hasattr(model, "models") or len(model.models) == 0:
            raise ValueError("IntegratedAcquisitionFunction requires at least one model to integrate!")
        if len(self._functions) == 0 or len(self._functions) != len(model.models):
            self._functions = [copy.deepcopy(self.acquisition_function) for _ in model.models]
        for submodel, func in zip(model.models, self._functions):
            func.update(model=submodel, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the EI value and its derivatives.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Expected Improvement of X
        """
        if self._functions is None:
            raise ValueError("Need to call update first!")
        return np.array([func._compute(X) for func in self._functions]).mean(axis=0)
