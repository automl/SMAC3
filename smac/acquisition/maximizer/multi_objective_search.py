from __future__ import annotations

from typing import Any

import itertools
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from pygmo import fast_non_dominated_sorting

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.local_and_random_search import (
    LocalAndSortedRandomSearch,
)
from smac.acquisition.maximizer.local_search import LocalSearch
from smac.utils.configspace import (
    convert_configurations_to_array,
    get_one_exchange_neighbourhood,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class MOLocalSearch(LocalSearch):
    def _get_initial_points(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        additional_start_points: list[tuple[float, Configuration]] | None,
    ) -> list[Configuration]:
        """Get initial points to start search from.

        If we already have a population, add those to the initial points.

        Parameters
        ----------
        previous_configs : list[Configuration]
            Previous configuration (e.g., from the runhistory).
        n_points : int
            Number of initial points to be generated.
        additional_start_points : list[tuple[float, Configuration]] | None
            Additional starting points.

        Returns
        -------
        list[Configuration]
            A list of initial points/configurations.
        """
        init_points = super()._get_initial_points(
            previous_configs=previous_configs, n_points=n_points, additional_start_points=additional_start_points
        )

        # Add population to Local search
        # TODO where is population saved? update accordingly
        if len(stats.population) > 0:
            population = [runhistory.ids_config[confid] for confid in stats.population]
            init_points = self._unique_list(itertools.chain(population, init_points))
        return init_points

    def _create_sort_keys(self, costs: np.array) -> list[list[float]]:
        """Non-Dominated Sorting of Costs

         In case of the predictive model returning the prediction for more than one objective per configuration
        (for example multi-objective or EIPS) we sort here based on the dominance order. In each front
        configurations are sorted on the number of points they dominate overall.

        Parameters
        ----------
        costs : np.array
            Cost(s) per config

        Returns
        -------
        list[list[float]]
            Sorting sequence for lexsort
        """
        _, domination_list, _, non_domination_rank = fast_non_dominated_sorting(costs)
        domination_list = [len(i) for i in domination_list]
        sort_objectives = [domination_list, non_domination_rank]  # Last column is primary sort key!
        return sort_objectives


class MOLocalAndSortedRandomSearch(LocalAndSortedRandomSearch):
    """Local and Random Search for Multi-Objective

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
        [LocalSearch] number of steps during a plateau walk before local search terminates.
    local_search_iterations: int, defauts to 10
        [Local Search] number of local search iterations.
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
            configspace=configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            local_search_iterations=local_search_iterations,
            seed=seed,
        )

        self.local_search = MOLocalSearch(
            configspace=configspace,
            acquisition_function=acquisition_function,
            challengers=challengers,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk,
            seed=seed,
        )
