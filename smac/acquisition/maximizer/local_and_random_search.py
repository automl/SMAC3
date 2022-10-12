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
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points according, to the acquisition function, uses the
    acquisition function to sort randomly sampled configurations. Random configurations are interleaved by the main SMAC
    code.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates
    local_search_iterations: int, defauts to 10
        [Local Search] number of local search iterations
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
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            seed=seed,
        )

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

    @property
    def acquisition_function(self) -> AbstractAcquisitionFunction | None:  # noqa: D102
        """Returns the used acquisition function."""
        return self._acquisition_function

    @acquisition_function.setter
    def acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self._acquisition_function = acquisition_function
        self._random_search._acquisition_function = acquisition_function
        self._local_search._acquisition_function = acquisition_function

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "random_search": self._random_search.meta,
                "local_search": self._local_search.meta,
            }
        )

        return meta

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self._random_search._maximize(
            previous_configs=previous_configs,
            n_points=n_points,
            _sorted=True,
        )

        next_configs_by_local_search = self._local_search._maximize(
            previous_configs=previous_configs,
            n_points=self._local_search_iterations,
            additional_start_points=next_configs_by_random_search_sorted,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = next_configs_by_random_search_sorted + next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        first_five = [f"{_[0]} ({_[1].origin})" for _ in next_configs_by_acq_value[:5]]

        logger.debug(
            "First 5 acquisition function values of selected configurations:\n%s",
            ", ".join(first_five),
        )

        return next_configs_by_acq_value


class LocalAndSortedPriorRandomSearch(AbstractAcquisitionMaximizer):
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points according to the acquisition function, uses the
    acquisition function to sort randomly sampled configurations. Random configurations are interleaved by the main SMAC
    code. The random configurations are retrieved from two different ConfigurationSpaces - one which uses priors
    (e.g. NormalFloatHP) and is defined by the user, and one that is a uniform version of the same space, i.e. with the
    priors removed.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The original ConfigurationSpace specified by the user.
    uniform_configspace : ConfigurationSpace
        A version of the user-defined ConfigurationSpace where all parameters are uniform (or have their weights removed
        in the case of a categorical hyperparameter).
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates.
    local_search_iterations: int, defaults to 10
        [Local Search] number of local search iterations.
    prior_sampling_fraction: float, defaults to 0.5
        The ratio of random samples that are taken from the user-defined ConfigurationSpace, as opposed to the uniform
        version.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        uniform_configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        prior_sampling_fraction: float = 0.5,
        seed: int = 0,
    ) -> None:
        super().__init__(
            acquisition_function,
            configspace,
            challengers=challengers,
            seed=seed,
        )

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

        self._local_search = LocalSearch(
            acquisition_function=acquisition_function,
            configspace=configspace,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )

        self._local_search_iterations = local_search_iterations
        self._prior_sampling_fraction = prior_sampling_fraction

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
    ) -> list[tuple[float, Configuration]]:

        # Get configurations sorted by EI
        next_configs_by_prior_random_search_sorted = self._prior_random_search._maximize(
            previous_configs,
            round(n_points * self._prior_sampling_fraction),
            _sorted=True,
        )

        # Get configurations sorted by EI
        next_configs_by_uniform_random_search_sorted = self._uniform_random_search._maximize(
            previous_configs,
            round(n_points * (1 - self._prior_sampling_fraction)),
            _sorted=True,
        )
        next_configs_by_random_search_sorted = []
        next_configs_by_random_search_sorted.extend(next_configs_by_prior_random_search_sorted)
        next_configs_by_random_search_sorted.extend(next_configs_by_uniform_random_search_sorted)

        next_configs_by_local_search = self._local_search._maximize(
            previous_configs,
            self._local_search_iterations,
            additional_start_points=next_configs_by_random_search_sorted,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = next_configs_by_random_search_sorted + next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        logger.debug(
            "First 5 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:5]]),
        )

        return next_configs_by_acq_value
