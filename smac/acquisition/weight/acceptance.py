from __future__ import annotations

from abc import abstractmethod
from typing import Any

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function.abstract_acquisition_function import (
    AbstractAcquisitionFunction,
)
from smac.acquisition.function.confidence_bound import LCB
from smac.acquisition.weight.prior import PriorWeight
from smac.model.abstract_model import AbstractModel
from smac.runhistory.runhistory import RunHistory
from smac.utils.configspace import create_prior_configspace_copy
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class AbstractPriorAcceptancePolicy:
    """Decides whether a user belief about the optimum is plausible enough to act on.

    A belief that points somewhere bad costs trials, and the surrogate model usually has an opinion about the
    region the belief names. Checking that opinion before acting is what keeps a mistaken belief from steering
    the search, without taking the decision away from the user: the default policy accepts everything.

    A policy must accept when there is nothing to judge on - no model, or no finished trials. Rejecting a belief
    because nothing is known yet would reject exactly the beliefs stated at the start of a run, which are the
    ones with the most to contribute.
    """

    @property
    def meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {"name": self.__class__.__name__}

    @abstractmethod
    def accept(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
    ) -> bool:
        """Whether the belief should be acted on.

        Parameters
        ----------
        prior : PriorWeight
            The belief, already anchored.
        configspace : ConfigurationSpace
            The search space.
        model : AbstractModel | None
            The surrogate model of the objective, or `None` if it has not been fitted yet.
        runhistory : RunHistory | None
            What has been observed so far.
        incumbent : Configuration | None
            The best configuration so far, or `None` if there is none yet.
        rng : np.random.RandomState | None, defaults to None
            Random state to draw samples with.
        """
        raise NotImplementedError


class AcceptAllPriors(AbstractPriorAcceptancePolicy):
    """Acts on every belief. The default: the user is in charge."""

    def accept(self, prior: PriorWeight, **kwargs: Any) -> bool:  # noqa: D102
        return True


class IncumbentComparisonPolicy(AbstractPriorAcceptancePolicy):
    r"""Rejects a belief whose region the surrogate model thinks is worse than the incumbent's neighbourhood.

    Follows the safeguard of "Dynamic Priors in Bayesian Optimization for Hyperparameter Optimization" by Lukas
    Fehring et al. [[FWS+25][FWS+25]]:

    $$\mathbb{E}_{\lambda \sim \pi}[\xi(\lambda)] - \mathbb{E}_{\lambda \sim N_{\hat{\lambda}}}[\xi(\lambda)] > \tau$$

    Configurations are drawn from the belief and from a belief-shaped neighbourhood of the current incumbent,
    both are scored with $\xi$ under the current surrogate, and the belief is accepted when its mean score beats
    the incumbent neighbourhood's by more than $\tau$.

    Note the reference distribution is the incumbent's neighbourhood, not the search space as a whole. Comparing
    against the whole space would accept almost any belief once the search has narrowed, since most of the space
    is worse than anywhere the search is still looking.

    Note also that the threshold is in raw objective units, so it has to be set for the scale of the objective
    at hand. The default comes from a benchmark whose objective lives in [0, 1].

    Parameters
    ----------
    n_samples : int, defaults to 100
        Number of configurations drawn from each distribution.
    threshold : float, defaults to -0.15
        How much worse than the incumbent's neighbourhood the belief's region may look and still be accepted.
        Negative, so a belief is rejected only when the model is fairly confident it is bad.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
        Used to score the samples. `None` is a lower confidence bound, which judges a region by what it might
        deliver rather than only by its mean, so an unexplored region is not rejected for being unexplored.
    std_denominator : float, defaults to 4.0
        Width of the incumbent's neighbourhood, as a divisor of the range of each hyperparameter.
    """

    def __init__(
        self,
        *,
        n_samples: int = 100,
        threshold: float = -0.15,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        std_denominator: float = 4.0,
    ) -> None:
        if n_samples < 1:
            raise ValueError(f"At least one sample is needed, got {n_samples}.")

        self._n_samples = n_samples
        self._threshold = threshold
        self._acquisition_function = acquisition_function if acquisition_function is not None else LCB()
        self._std_denominator = std_denominator

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "n_samples": self._n_samples,
                "threshold": self._threshold,
                "acquisition_function": self._acquisition_function.meta,
                "std_denominator": self._std_denominator,
            }
        )

        return meta

    def accept(
        self,
        prior: PriorWeight,
        *,
        configspace: ConfigurationSpace,
        model: AbstractModel | None,
        runhistory: RunHistory | None,
        incumbent: Configuration | None,
        rng: np.random.RandomState | None = None,
    ) -> bool:
        """Whether the surrogate model thinks the belief's region is worth looking at."""
        if model is None or incumbent is None or runhistory is None or len(runhistory) == 0:
            logger.debug("Nothing has been observed yet, so there is no ground to reject a prior on.")

            return True

        if prior.prior is None:
            return True

        prior_samples = prior.prior.sample(self._n_samples, rng)
        if prior_samples is None:
            logger.debug("The prior cannot be sampled from, so its region cannot be judged.")

            return True

        neighbourhood = create_prior_configspace_copy(
            configspace,
            dict(incumbent),
            std_denominator=self._std_denominator,
        )
        incumbent_samples = list(neighbourhood.sample_configuration(size=self._n_samples))

        self._acquisition_function.update(model=model, num_data=len(runhistory), eta=0.0)

        prior_value = float(np.mean(self._acquisition_function(prior_samples)))
        incumbent_value = float(np.mean(self._acquisition_function(incumbent_samples)))
        difference = prior_value - incumbent_value

        if difference > self._threshold:
            return True

        logger.info(
            f"Rejecting a user prior: the surrogate scores its region {-difference:.4f} below the incumbent's "  # noqa: E231,E501
            f"neighbourhood, past the threshold of {-self._threshold:.4f}."  # noqa: E231
        )

        return False
