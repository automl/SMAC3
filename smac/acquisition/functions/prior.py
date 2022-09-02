from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter

from smac.acquisition.functions.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.utils.logging import get_logger
from ConfigSpace import Configuration
from smac.model.abstract_model import AbstractModel
from smac.acquisition.functions.integrated import IntegratedAcquisitionFunction
from smac.acquisition.functions.thompson import TS
from smac.acquisition.functions.confidence_bound import LCB


__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PriorAcquisitionFunction(AbstractAcquisitionFunction):
    r"""Weight the acquisition function with a user-defined prior over the optimum.

    See "piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization" by Carl
    Hvarfner et al. [1]_ for further details.

    References
    ----------
    .. [1] [piBO, Hvarfner et al., 2022](https://arxiv.org/pdf/2204.11051.pdf)

    Parameters
    ----------
    decay_beta: float
        Decay factor on the user prior - defaults to n_iterations / 10 if not specifed
        otherwise.
    prior_floor : float, defaults to 1e-12
        Lowest possible value of the prior, to ensure non-negativity for all values
        in the search space.
    discretize : bool, defaults to False
        Whether to discretize (bin) the densities for continous parameters. Triggered
        for Random Forest models and continous hyperparameters to avoid a pathological case
        where all Random Forest randomness is removed (RF surrogates require piecewise constant
        acquisition functions to be well-behaved).
    discrete_bins_factor : float, defaults to 10.0
        If discretizing, the multiple on the number of allowed bins for
        each parameter.

    kwargs : Any
        Additional keyword arguments
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
        self.long_name = "Prior Acquisition Function (%s)" % acquisition_function.__class__.__name__
        self.acq: AbstractAcquisitionFunction = acquisition_function
        self._functions: list[AbstractAcquisitionFunction] = []
        self.eta: float | None = None

        self._hyperparameters: dict[Any, list[Configuration]] | None = None
        self.decay_beta = decay_beta
        self.prior_floor = prior_floor
        self.discretize = discretize
        if self.discretize:
            self.discrete_bins_factor = discrete_bins_factor

        # check if the acquisition function is LCB or TS - then the acquisition function values
        # need to be rescaled to assure positiveness & correct magnitude
        if isinstance(self.acq, IntegratedAcquisitionFunction):
            acquisition_type = self.acq.acq
        else:
            acquisition_type = self.acq

        self.rescale_acq = isinstance(acquisition_type, (LCB, TS))
        self.iteration_number = 0

    @property
    def hyperparameters(self):
        if self._hyperparameters is None:
            raise ValueError("Please set the model via '_set_model' first.")
        return self._hyperparameters

    @hyperparameters.setter
    def hyperparameters(self, hyperparameters):
        self._hyperparameters = hyperparameters

    def _set_model(self, model: AbstractModel) -> None:
        self.model = model
        self.hyperparameters = self.model.get_configspace().get_hyperparameters_dict()

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def update(self, eta: float, **kwargs: Any) -> None:
        """Update the acquisition function attributes required for calculation.

        Parameters
        ----------
        eta : float
            Current incumbent value.
        """
        self.iteration_number += 1
        self.eta = eta
        # Maybe the underlying acquisition function needs eta.
        kwargs["eta"] = eta
        self.acq.update(**kwargs)

    def _compute_prior(self, X: np.ndarray) -> np.ndarray:
        """Computes the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the user-specified prior
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            The user prior over the optimum for values of X
        """
        prior_values = np.ones((len(X), 1))
        # iterate over the hyperparmeters (alphabetically sorted) and the columns, which come
        # in the same order
        for parameter, X_col in zip(self.hyperparameters.values(), X.T):
            if self.discretize and isinstance(parameter, FloatHyperparameter):
                number_of_bins = int(np.ceil(self.discrete_bins_factor * self.decay_beta / self.iteration_number))
                prior_values *= self._compute_discretized_pdf(parameter, X_col, number_of_bins) + self.prior_floor
            else:
                prior_values *= parameter._pdf(X_col[:, np.newaxis])

        return prior_values

    def _compute_discretized_pdf(
        self, parameter: FloatHyperparameter, X_col: np.ndarray, number_of_bins: int
    ) -> np.ndarray:
        """Discretizes (bins) prior values on continous a specific continous parameter
        to an increasingly coarse discretization determined by the prior decay parameter.

        Parameters
        ----------
        parameter: a FloatHyperparameter that, due to using a random forest
            surrogate, must have its prior discretized
        X_col: np.ndarray(N, ), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, ), with N as
            the number of points to evaluate for the specific hyperparameter
        number_of_bins: The number of unique values allowed on the
            discretized version of the pdf.

        Returns
        -------
        np.ndarray(N,1)
            The user prior over the optimum for the parameter at hand.
        """
        # evaluates the actual pdf on all the relevant points
        pdf_values = parameter._pdf(X_col[:, np.newaxis])
        # retrieves the largest value of the pdf in the domain
        lower, upper = (0, parameter.get_max_density())
        # creates the bins (the possible discrete options of the pdf)
        bin_values = np.linspace(lower, upper, number_of_bins)
        # generates an index (bin) for each evaluated point
        bin_indices = np.clip(
            np.round((pdf_values - lower) * number_of_bins / (upper - lower)), 0, number_of_bins - 1
        ).astype(int)
        # gets the actual value for each point
        prior_values = bin_values[bin_indices]
        return prior_values

    def _compute(self, X: np.ndarray) -> np.ndarray:
        """Computes the prior-weighted acquisition function values, where the prior on each
        parameter is multiplied by a decay factor controlled by the parameter decay_beta and
        the iteration number. Multivariate priors are not supported, for now.

        Parameters
        ----------
        X: np.ndarray(N, D), The input points where the acquisition function
            should be evaluated. The dimensionality of X is (N, D), with N as
            the number of points to evaluate at and D is the number of
            dimensions of one X.

        Returns
        -------
        np.ndarray(N,1)
            Prior-weighted acquisition function values of X
        """
        if self.rescale_acq:
            # for TS and UCB, we need to scale the function values to not run into issues
            # of negative values or issues of varying magnitudes (here, they are both)
            # negative by design and just flipping the sign leads to picking the worst point)
            acq_values = np.clip(self.acq._compute(X) + self.eta, 0, np.inf)
        else:
            acq_values = self.acq._compute(X)
        prior_values = self._compute_prior(X) + self.prior_floor
        decayed_prior_values = np.power(prior_values, self.decay_beta / self.iteration_number)

        return acq_values * decayed_prior_values
