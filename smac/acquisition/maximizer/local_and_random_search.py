from __future__ import annotations

from typing import Any

from ConfigSpace import Configuration, ConfigurationSpace

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.acquisition.maximizer.local_search import LocalSearch
from smac.acquisition.maximizer.random_search import RandomSearch
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class LocalAndSortedRandomSearch(AbstractAcquisitionMaximizer):
    """Implement SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points according to the acquisition
    function, uses the acquisition function to sort randomly sampled configurations.
    Random configurations are interleaved by the main SMAC code.

    The Random configurations are interleaved to circumvent issues from a constant prediction
    from the Random Forest model at the beginning of the optimization process.

    Parameters
    ----------
    configspace : ConfigurationSpace
    uniform_configspace : ConfigurationSpace
        A version of the user-defined ConfigurationSpace where all parameters are uniform (or have their weights removed
        in the case of a categorical hyperparameter). Can optionally be given and sampling ratios be defined via the
        `prior_sampling_fraction` parameter.
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates.
    local_search_iterations: int, defauts to 10
        [Local Search] number of local search iterations.
    prior_sampling_fraction: float, defaults to 0.5
        The ratio of random samples that are taken from the user-defined ConfigurationSpace, as opposed to the uniform
        version (needs `uniform_configspace`to be defined).
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        seed: int = 0,
        uniform_configspace: ConfigurationSpace | None = None,
        prior_sampling_fraction: float | None = None,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        if uniform_configspace is not None and prior_sampling_fraction is None:
            prior_sampling_fraction = 0.5
        if uniform_configspace is None and prior_sampling_fraction is not None:
            raise ValueError("If `prior_sampling_fraction` is given, `uniform_configspace` must be defined.")
        if uniform_configspace is not None and prior_sampling_fraction is not None:
            self._prior_random_search = RandomSearch(
                acquisition_function=acquisition_function,
                configspace=configspace,
                seed=seed,
            )

            self._uniform_random_search = RandomSearch(
                acquisition_function=acquisition_function,
                configspace=uniform_configspace,
                seed=seed,
            )
        else:
            self._random_search = RandomSearch(
                configspace=configspace,
                acquisition_function=acquisition_function,
                seed=seed,
            )

        self._local_search = LocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )

        self._local_search_iterations = local_search_iterations
        self._prior_sampling_fraction = prior_sampling_fraction
        self._uniform_configspace = uniform_configspace

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        if self._uniform_configspace is not None:
            self._prior_random_search._acquisition_function = acquisition_function
            self._uniform_random_search._acquisition_function = acquisition_function
        else:
            self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        if self._uniform_configspace is None:
            meta.update(
                {
                    "random_search": self._random_search.meta,
                    "local_search": self._local_search.meta,
                }
            )
        else:
            meta.update(
                {
                    "prior_random_search": self._prior_random_search.meta,
                    "uniform_random_search": self._uniform_random_search.meta,
                    "local_search": self._local_search.meta,
                }
            )

        return meta

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:

        if self._uniform_configspace is not None and self._prior_sampling_fraction is not None:
            # Get configurations sorted by acquisition function value
            next_configs_by_prior_random_search_sorted = self._prior_random_search._maximize(
                previous_configs,
                round(n_points * self._prior_sampling_fraction),
                _sorted=True,
            )

            # Get configurations sorted by acquisition function value
            next_configs_by_uniform_random_search_sorted = self._uniform_random_search._maximize(
                previous_configs,
                round(n_points * (1 - self._prior_sampling_fraction)),
                _sorted=True,
            )
            next_configs_by_random_search_sorted = (
                next_configs_by_uniform_random_search_sorted + next_configs_by_prior_random_search_sorted
            )
            next_configs_by_random_search_sorted.sort(reverse=True, key=lambda x: x[0])
        else:
            # Get configurations sorted by acquisition function value
            next_configs_by_random_search_sorted = self._random_search._maximize(
                previous_configs=previous_configs,
                n_points=n_points,
                _sorted=True,
            )

        # Choose the best self._local_search_iterations random configs to start the local search, and choose only
        # incumbent from previous configs
        random_starting_points = next_configs_by_random_search_sorted[: self._local_search_iterations]
        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=random_starting_points,
        )

        next_configs_by_acq_value = next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        logger.debug(f"First 5 acquisition function values of selected configurations:\n{', '.join(first_five)}")

        return next_configs_by_acq_value
