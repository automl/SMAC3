from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace import Configuration
from ConfigSpace.hyperparameters import FloatHyperparameter

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import LCB
from smac.acquisition.function.integrated_acquisition_function import (
    IntegratedAcquisitionFunction,
)
from smac.acquisition.function.thompson import TS
from smac.model.abstract_model import AbstractModel
from smac.model.random_forest.abstract_random_forest import AbstractRandomForest
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PriorAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Weight the acquisition function with a user-defined prior over the optimum.

    See "piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization" by Carl
    Hvarfner et al. [HSSL22]_ for further details.

    Parameters
    ----------
    decay_beta: float
        Decay factor on the user prior. A solid default value for decay_beta (empirically founded) is
        ``scenario.n_trials`` / 10.
    prior_floor : float, defaults to 1e-12
        Lowest possible value of the prior to ensure non-negativity for all values in the search space.
    discretize : bool, defaults to False
        Whether to discretize (bin) the densities for continous parameters. Triggered for Random Forest models and
        continous hyperparameters to avoid a pathological case where all Random Forest randomness is removed
        (RF surrogates require piecewise constant acquisition functions to be well-behaved).
    discrete_bins_factor : float, defaults to 10.0
        If discretizing, the multiple on the number of allowed bins for each parameter.
    """

    def __init__(
        self,
        acquisition_function: AbstractAcquisitionFunction,
        decay_beta: float,
        prior_floor: float = 1e-12,
        discretize: bool = False,
        discrete_bins_factor: float = 10.0,
    ):
        super().__init__()
        self._acquisition_function: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self._eta: float | None = None

        self._hyperparameters: dict[Any, Configuration] | None = None
        self._decay_beta = decay_beta
        self._prior_floor = prior_floor
        self._discretize = discretize
        self._discrete_bins_factor = discrete_bins_factor

        # check if the acquisition function is LCB or TS - then the acquisition function values
        # need to be rescaled to assure positiveness & correct magnitude
        if isinstance(self._acquisition_function, IntegratedAcquisitionFunction):
            acquisition_type = self._acquisition_function._acquisition_function
        else:
            acquisition_type = self._acquisition_function

        self._rescale = isinstance(acquisition_type, (LCB, TS))
        self._iteration_number = 0

    @property
    def name(self) -> str:  # noqa: D102
        return f"Prior Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "acquisition_function": self._acquisition_function.meta,
                "decay_beta": self._decay_beta,
                "prior_floor": self._prior_floor,
                "discretize": self._discretize,
                "discrete_bins_factor": self._discrete_bins_factor,
            }
        )

        return meta

    @property
    def model(self) -> AbstractModel | None:  # noqa: D102
        return self._model

    @model.setter
    def model(self, model: AbstractModel) -> None:
        self._model = model
        self._hyperparameters = model._configspace.get_hyperparameters_dict()

        if isinstance(model, AbstractRandomForest):
            if not self._discretize:
                logger.warning("Discretizing the prior for random forest models.")
                self._discretize = True

    def _update(self, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        eta : float
            Current incumbent value.
        """
        assert "eta" in kwargs
        self._iteration_number += 1
        self._eta = kwargs["eta"]

        assert self.model is not None
        self._acquisition_function.update(model=self.model, **kwargs)

    def _compute_prior(self, X: np.ndarray) -> np.ndarray:
        """Compute the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray [N, D]
            The input points where the user-specified prior should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N, 1]
            The user prior over the optimum for values of X.
        """
        assert self._hyperparameters is not None

        prior_values = np.ones((len(X), 1))
        # iterate over the hyperparmeters (alphabetically sorted) and the columns, which come
        # in the same order
        for parameter, X_col in zip(self._hyperparameters.values(), X.T):
            if self._discretize and isinstance(parameter, FloatHyperparameter):
                assert self._discrete_bins_factor is not None
                number_of_bins = int(np.ceil(self._discrete_bins_factor * self._decay_beta / self._iteration_number))
                prior_values *= self._compute_discretized_pdf(parameter, X_col, number_of_bins) + self._prior_floor
            else:
                prior_values *= parameter._pdf(X_col[:, np.newaxis])

        return prior_values

    def _compute_discretized_pdf(
        self,
        hyperparameter: FloatHyperparameter,
        X_col: np.ndarray,
        number_of_bins: int,
    ) -> np.ndarray:
        """Discretize (bins) prior values on continous a specific continous parameter
        to an increasingly coarse discretization determined by the prior decay parameter.

        Parameters
        ----------
        hyperparameter : FloatHyperparameter
            A float hyperparameter that, due to using a random forest surrogate, must have its prior discretized.
        X_col: np.ndarray [N, ]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, ),
            with N as the number of points to evaluate for the specific hyperparameter.
        number_of_bins : int
            The number of unique values allowed on the discretized version of the pdf.

        Returns
        -------
        np.ndarray [N, 1]
            The user prior over the optimum for the parameter at hand.
        """
        # Evaluates the actual pdf on all the relevant points
        pdf_values = hyperparameter._pdf(X_col[:, np.newaxis])

        # Retrieves the largest value of the pdf in the domain
        lower, upper = (0, hyperparameter.get_max_density())

        # Creates the bins (the possible discrete options of the pdf)
        bin_values = np.linspace(lower, upper, number_of_bins)

        # Generates an index (bin) for each evaluated point
        bin_indices = np.clip(
            np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1
        ).astype(int)

        # Gets the actual value for each point
        prior_values = bin_values[bin_indices]

        return prior_values

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Compute the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray [N, D]
            The input points where the acquisition function should be evaluated. The dimensionality of X is (N, D),
            with N as the number of points to evaluate at and D is the number of dimensions of one X.

        Returns
        -------
        np.ndarray [N, 1]
            Prior-weighted acquisition function values of X
        """
        if self._rescale:
            # for TS and UCB, we need to scale the function values to not run into issues
            # of negative values or issues of varying magnitudes (here, they are both)
            # negative by design and just flipping the sign leads to picking the worst point)
            acq_values = np.clip(self._acquisition_function._compute(X) + self._eta, 0, np.inf)
        else:
            acq_values = self._acquisition_function._compute(X)

        prior_values = self._compute_prior(X) + self._prior_floor
        decayed_prior_values = np.power(prior_values, self._decay_beta / self._iteration_number)

        return acq_values * decayed_prior_values
