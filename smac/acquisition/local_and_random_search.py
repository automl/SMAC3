from __future__ import annotations

import abc
from typing import Any, Callable, Iterator, Set

import copy
import itertools
import logging
import time

import numpy as np

from smac.acquisition import AbstractAcquisitionOptimizer
from smac.acquisition.functions import AbstractAcquisitionFunction
from smac.acquisition.local_search import LocalSearch
from smac.acquisition.random_search import RandomSearch

from ConfigSpace import Configuration, ConfigurationSpace


from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class LocalAndSortedRandomSearch(AbstractAcquisitionOptimizer):
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations. Random configurations are
    interleaved by the main SMAC code.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction | None, defaults to None
    max_steps: int | None, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates
    local_search_iterations: int, defauts to 0
        [Local Search] number of local search iterations
    challengers : int, defaults to 5000
        Number of challengers.
    seed : int, defaults to 0

    Attributes
    ----------
    random_search : RandomSearch
    local_search : LocalSearch
    local_search_iterations : int
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        challengers: int = 5000,
        seed: int = 0,
    ) -> None:
        super().__init__(configspace, acquisition_function=acquisition_function, challengers=challengers, seed=seed)
        self.random_search = RandomSearch(configspace=configspace, acquisition_function=acquisition_function, seed=seed)
        self.local_search = LocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )
        self.local_search_iterations = local_search_iterations

    def get_meta(self) -> dict[str, Any]:
        """Returns the meta data of the created object."""
        return {
            "name": self.__class__.__name__,
        }

    def _set_acquisition_function(self, acquisition_function: AbstractAcquisitionFunction) -> None:
        self.acquisition_function = acquisition_function
        self.random_search._set_acquisition_function(acquisition_function)
        self.local_search._set_acquisition_function(acquisition_function)

    def _maximize(
        self,
        previous_configs: list[Configuration],
        num_points: int,
    ) -> list[tuple[float, Configuration]]:

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            previous_configs,
            num_points,
            _sorted=True,
        )

        next_configs_by_local_search = self.local_search._maximize(
            previous_configs,
            self.local_search_iterations,
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


class LocalAndSortedPriorRandomSearch(AbstractAcquisitionOptimizer):
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points
    according to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations. Random configurations are
    interleaved by the main SMAC code. The random configurations are retrieved
    from two different ConfigurationSpaces - one which uses priors (e.g. NormalFloatHP)
    and is defined by the user, and one that is a uniform version of the same
    space, i.e. with the priors removed.

    Parameters
    ----------
    configspace : ConfigurationSpace
        The original ConfigurationSpace specified by the user
    acquisition_function : AbstractAcquisitionFunction
    uniform_configspace : ConfigurationSpace
        A version of the user-defined ConfigurationSpace where all parameters are
        uniform (or have their weights removed in the case of a categorical
        hyperparameter)
    max_steps: int, defaults to None
        [LocalSearch] Maximum number of steps that the local search will perform
    n_steps_plateau_walk: int, defaults to 10
        [LocalSearch] number of steps during a plateau walk before local search terminates
    local_search_iterations: int, defaults to 10
        [Local Search] number of local search iterations
    prior_sampling_fraction: float, defaults to 0.5
        The ratio of random samples that are taken from the user-defined ConfigurationSpace,
        as opposed to the uniform version.
    challengers : int, defaults to 5000
        Number of challengers.
    seed : int, defaults to 0

    Attributes
    ----------
    prior_random_search : RandomSearch
        Search over the prior configuration space.
    uniform_random_search : RandomSearch
        Search ove rthe normal configuration space.
    local_search : LocalSearch
    local_search_iterations : int
    prior_sampling_fraction : float
        The ratio of random samples that are taken from the user-defined ConfigurationSpace,
        as opposed to the uniform version.
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction,
        uniform_configspace: ConfigurationSpace,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        local_search_iterations: int = 10,
        prior_sampling_fraction: float = 0.5,
        challengers: int = 5000,
        seed: int = 0,
    ) -> None:
        super().__init__(acquisition_function, configspace, challengers=challengers, seed=seed)
        self.prior_random_search = RandomSearch(
            acquisition_function=acquisition_function, configspace=configspace, seed=seed
        )
        self.uniform_random_search = RandomSearch(
            acquisition_function=acquisition_function, configspace=uniform_configspace, seed=seed
        )
        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            configspace=configspace,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )
        self.local_search_iterations = local_search_iterations
        self.prior_sampling_fraction = prior_sampling_fraction

    def _maximize(
        self,
        previous_configs: list[Configuration],
        num_points: int,
    ) -> list[tuple[float, Configuration]]:

        # Get configurations sorted by EI
        next_configs_by_prior_random_search_sorted = self.prior_random_search._maximize(
            previous_configs,
            round(num_points * self.prior_sampling_fraction),
            _sorted=True,
        )

        # Get configurations sorted by EI
        next_configs_by_uniform_random_search_sorted = self.uniform_random_search._maximize(
            previous_configs,
            round(num_points * (1 - self.prior_sampling_fraction)),
            _sorted=True,
        )
        next_configs_by_random_search_sorted = []
        next_configs_by_random_search_sorted.extend(next_configs_by_prior_random_search_sorted)
        next_configs_by_random_search_sorted.extend(next_configs_by_uniform_random_search_sorted)

        next_configs_by_local_search = self.local_search._maximize(
            previous_configs,
            self.local_search_iterations,
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
