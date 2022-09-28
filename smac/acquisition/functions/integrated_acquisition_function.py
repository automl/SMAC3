from __future__ import annotations

from typing import Any

import copy

import numpy as np

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.model.abstract_model import AbstractModel
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class IntegratedAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Marginalizes over model hyperparameters to compute the integrated acquisition function.

    See "Practical Bayesian Optimization of Machine Learning Algorithms" by Jasper Snoek et al.
    (https://papers.nips.cc/paper/4522-practical-bayesian-optimization-of-machine-learning-algorithms.pdf)
    for further details.

    Parameters
    ----------
    acquisition_function : AbstractAcquisitionFunction
        The acquisition function, which should be integrated.
    """

    def __init__(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        super().__init__()
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self._eta: float | None = None

    @property
    def name(self) -> str:  # noqa: D102
        return f"Integrated Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update({"acquisition_function": self._acquisition_function.meta})

        return meta

    def _update(self, **kwargs: Any) -> None:
        """Updates the acquisition functions values.

        This method will be called if the model is updated. For example, entropy search uses it to update its
        approximation of P(x=x_min) and EI uses it to update the current fmin.

        This implementation creates an acquisition function object for each model to integrate over and sets the
        respective attributes for each acquisition function object.

        Parameters
        ----------
        kwargs : Any
            Keyword arguments for the model.
        """
        model = self.model
        models: list[AbstractModel] | None = None
        if hasattr(model, "models"):
            models = model.models  # type: ignore

        if models is None or len(models) == 0:
            raise ValueError("IntegratedAcquisitionFunction requires at least one model to integrate!")

        if len(self._functions) == 0 or len(self._functions) != len(models):
            self._functions = [copy.deepcopy(self._acquisition_function) for _ in models]

        for submodel, func in zip(models, self._functions):
            func.update(model=submodel, **kwargs)

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computess the EI value and its derivatives."""
        if self._functions is None:
            raise ValueError("Need to call `update` first!")

        return np.array([func._compute(X) for func in self._functions]).mean(axis=0)
