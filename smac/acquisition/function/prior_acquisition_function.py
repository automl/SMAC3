from __future__ import annotations

from typing import Any

import numpy as np
from ConfigSpace.hyperparameters import FloatHyperparameter

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.weighted_acquisition_function import (
    WeightedAcquisitionFunction,
)
from smac.acquisition.weight.composite import PriorEnsemble
from smac.acquisition.weight.decay import PolynomialDecay
from smac.acquisition.weight.prior import PriorWeight, discretize_pdf
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class PriorAcquisitionFunction(WeightedAcquisitionFunction):
    r"""Weight the acquisition function with a user-defined prior over the optimum.

    See "piBO: Augmenting Acquisition Functions with User Beliefs for Bayesian Optimization" by Carl
    Hvarfner et al. [[HSSL22][HSSL22]] for further details.

    This is a `WeightedAcquisitionFunction` carrying a single `PriorWeight`, which is where the mechanism lives
    and where it is documented. Reach for `WeightedAcquisitionFunction` directly when the acquisition function
    should carry more than one prior, or a prior alongside output constraints.

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
        prior = PriorWeight(
            decay=PolynomialDecay(beta=decay_beta),
            floor=prior_floor,
            # False here means "decide by the surrogate model", which is what this class has always done.
            discretize=True if discretize else None,
            discrete_bins_factor=discrete_bins_factor,
        )

        super().__init__(acquisition_function, [PriorEnsemble([prior])])

        self._prior = prior
        self._decay_beta = decay_beta

    @property
    def name(self) -> str:  # noqa: D102
        return f"Prior Acquisition Function ({self._acquisition_function.__class__.__name__})"

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        # Deliberately not the metadata of the weighted acquisition function: this dictionary ends up in the name
        # of the output directory, and changing it would keep existing runs from being continued.
        return {
            "name": self.__class__.__name__,
            "acquisition_function": self._acquisition_function.meta,
            "decay_beta": self._decay_beta,
            "prior_floor": self._prior_floor,
            "discretize": self._discretize,
            "discrete_bins_factor": self._discrete_bins_factor,
        }

    @property
    def prior(self) -> PriorWeight:
        """The weight carrying the user prior."""
        return self._prior

    @property
    def _prior_floor(self) -> float:
        return self._prior._floor

    @property
    def _discretize(self) -> bool:
        return bool(self._prior._discretize)

    @property
    def _discrete_bins_factor(self) -> float:
        return self._prior._discrete_bins_factor

    @property
    def _initial_design_size(self) -> int | None:
        return self._prior.t0

    @property
    def _iteration_number(self) -> int:
        return self._prior.steps

    @property
    def _hyperparameters(self) -> dict[str, Any] | None:
        if self._prior.prior is None:
            return None

        return {name: None for name in self._prior.prior.hyperparameter_names}

    def _compute_prior(self, X: np.ndarray) -> np.ndarray:
        """The prior density at X, before the floor and the decay are applied."""
        return self._prior._compute(X)

    def _compute_discretized_pdf(
        self,
        hyperparameter: FloatHyperparameter,
        X_col: np.ndarray,
        number_of_bins: int,
    ) -> np.ndarray:
        """Coarsens the density of a continuous hyperparameter. See `smac.acquisition.weight.prior`."""
        return discretize_pdf(hyperparameter, X_col, number_of_bins)
