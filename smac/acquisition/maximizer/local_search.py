from __future__ import annotations

from typing import Any

import itertools
import time

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acqusition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.utils.configspace import (
    convert_configurations_to_array,
    get_one_exchange_neighbourhood,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2022, automl.org"
__license__ = "3-clause BSD"

logger = get_logger(__name__)


class LocalSearch(AbstractAcquisitionMaximizer):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    configspace : ConfigurationSpace
    acquisition_function : AbstractAcquisitionFunction
    challengers : int, defaults to 5000
        Number of challengers.
    max_steps: int | None, defaults to None
        Maximum number of iterations that the local search will perform.
    n_steps_plateau_walk: int, defaults to 10
        Number of steps during a plateau walk before local search terminates.
    vectorization_min_obtain : int, defaults to 2
        Minimal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC.
    vectorization_max_obtain : int, defaults to 64
        Maximal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC.
    seed : int, defaults to 0
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        challengers: int = 5000,
        max_steps: int | None = None,
        n_steps_plateau_walk: int = 10,
        vectorization_min_obtain: int = 2,
        vectorization_max_obtain: int = 64,
        seed: int = 0,
    ) -> None:
        super().__init__(
            configspace,
            acquisition_function,
            challengers=challengers,
            seed=seed,
        )

        self._max_steps = max_steps
        self._n_steps_plateau_walk = n_steps_plateau_walk
        self._vectorization_min_obtain = vectorization_min_obtain
        self._vectorization_max_obtain = vectorization_max_obtain

    @property
    def meta(self) -> dict[str, Any]:  # noqa: D102
        meta = super().meta
        meta.update(
            {
                "max_steps": self._max_steps,
                "n_steps_plateau_walk": self._n_steps_plateau_walk,
                "vectorization_min_obtain": self._vectorization_min_obtain,
                "vectorization_max_obtain": self._vectorization_max_obtain,
            }
        )

        return meta

    def _maximize(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        additional_start_points: list[tuple[float, Configuration]] | None = None,
    ) -> list[tuple[float, Configuration]]:
        """Start a local search from the given start points. Iteratively collect neighbours
        using Configspace.utils.get_one_exchange_neighbourhood and evaluate them.
        If the new config is better than the current best, the local search is continued from the
        new config.

        Quit if either the max number of steps is reached or
        no neighbor with a higher improvement was found or the number of local steps self._n_steps_plateau_walk
        for each of the starting point is depleted.


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
            Final candidates.
        """
        init_points = self._get_initial_points(previous_configs, n_points, additional_start_points)
        configs_acq = self._search(init_points)

        # Shuffle for random tie-break
        self._rng.shuffle(configs_acq)

        # Sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        for a, inc in configs_acq:
            inc.origin = "Acquisition Function Maximizer: Local Search"

        return configs_acq

    def _get_initial_points(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        additional_start_points: list[tuple[float, Configuration]] | None,
    ) -> list[Configuration]:
        """Get initial points to start search from.

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
        sampled_points = []
        init_points = []
        n_init_points = n_points
        if len(previous_configs) < n_points:
            sampled_points = self._configspace.sample_configuration(size=n_points - len(previous_configs))
            n_init_points = len(previous_configs)
            if not isinstance(sampled_points, list):
                sampled_points = [sampled_points]
        if len(previous_configs) > 0:
            init_points = self._get_init_points_from_previous_configs(
                previous_configs,
                n_init_points,
                additional_start_points,
            )

        return sampled_points + init_points

    def _get_init_points_from_previous_configs(
        self,
        previous_configs: list[Configuration],
        n_points: int,
        additional_start_points: list[tuple[float, Configuration]] | None,
    ) -> list[Configuration]:
        """
        Generate a set of initial points from the previous configurations and possibly additional points.

        The idea is to decouple runhistory from the local search model and replace it with a more general
        form (list[Configuration]). This is useful to more quickly collect new configurations
        along the iterations, rather than feeding it to the runhistory every time.

        create three lists and concatenate them:
        1. sorted the previous configs by acquisition value
        2. sorted the previous configs by marginal predictive costs
        3. additional start points

        and create a list that carries unique configurations only. Crucially,
        when reading from left to right, all but the first occurrence of a configuration
        are dropped.

        Parameters
        ----------
        previous_configs: list[Configuration]
            Previous configuration (e.g., from the runhistory).
        n_points: int
            Number of initial points to be generated; selected from previous configs (+ random configs if necessary).
        additional_start_points: list[tuple[float, Configuration]] | None
            Additional starting points.

        Returns
        -------
        init_points: list[Configuration]
            A list of initial points.
        """
        assert self._acquisition_function is not None

        # configurations with the lowest predictive cost, check for None to make unit tests work
        if self._acquisition_function.model is not None:
            conf_array = convert_configurations_to_array(previous_configs)
            costs = self._acquisition_function.model.predict_marginalized(conf_array)[0]
            assert len(conf_array) == len(costs), (conf_array.shape, costs.shape)

            # In case of the predictive model returning the prediction for more than one objective per configuration
            # (for example multi-objective or EIPS) it is not immediately clear how to sort according to the cost
            # of a configuration. Therefore, we simply follow the ParEGO approach and use a random scalarization.
            if len(costs.shape) == 2 and costs.shape[1] > 1:
                weights = np.array([self._rng.rand() for _ in range(costs.shape[1])])
                weights = weights / np.sum(weights)
                costs = costs @ weights

            # From here: make argsort result to be random between equal values
            # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            random = self._rng.rand(len(costs))
            indices = np.lexsort((random.flatten(), costs.flatten()))  # Last column is primary sort key!

            # Cannot use zip here because the indices array cannot index the
            # rand_configs list, because the second is a pure python list
            previous_configs_sorted_by_cost = [previous_configs[ind] for ind in indices][:n_points]
        else:
            previous_configs_sorted_by_cost = []

        if additional_start_points is not None:
            additional_start_points = [asp[1] for asp in additional_start_points]
        else:
            additional_start_points = []

        init_points = []
        init_points_as_set: set[Configuration] = set()
        for cand in itertools.chain(
            previous_configs_sorted_by_cost,
            additional_start_points,
        ):
            if cand not in init_points_as_set:
                init_points.append(cand)
                init_points_as_set.add(cand)

        return init_points

    def _search(
        self,
        start_points: list[Configuration],
    ) -> list[tuple[float, Configuration]]:
        """Optimize the acquisition function.

        Execution:
        1. Neighbour generation strategy for each of the starting points is according to
        ConfigSpace.utils.get_one_exchange_neighbourhood.
        2. Each of the starting points create a local search, that can be active.
        if it is active, request a neighbour of its neightbourhood and evaluate it.
        3. Comparing the acquisition function of the neighbors with the acquisition value of the
        candidate.
        If it improved, then the candidate is replaced by the neighbor. And this candidate is
        investigated again with two new neighbours.
        If it did not improve, it is investigated with twice as many new neighbours
        (at most self._vectorization_max_obtain neighbours).
        The local search for a starting point is stopped if the number of evaluations is larger
        than self._n_steps_plateau_walk.


        Parameters
        ----------
        start_points : list[Configuration]
            Starting points for the search.

        Returns
        -------
        list[tuple[float, Configuration]]
            Candidates with their acquisition function value. (acq value, candidate)
        """
        assert self._acquisition_function is not None

        # Gather data structure for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]

        candidates = start_points
        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates_ = self._acquisition_function(candidates)

        if num_candidates == 1:
            acq_val_candidates = [acq_val_candidates_[0][0]]
        else:
            acq_val_candidates = [a[0] for a in acq_val_candidates_]

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_candidates
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_candidates
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_candidates
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_candidates
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_candidates
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = [self._vectorization_min_obtain] * num_candidates
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(candidates):
            neighborhood_iterators.append(
                # get_one_exchange_neighbourhood implementational details:
                # https://github.com/automl/ConfigSpace/blob/05ab3da2a06c084ba920e8e4e3f62f2e87e81442/ConfigSpace/util.pyx#L95
                # Return all configurations in a one-exchange neighborhood.
                #
                #     The method is implemented as defined by:
                #     Frank Hutter, Holger H. Hoos and Kevin Leyton-Brown
                #     Sequential Model-Based Optimization for General Algorithm Configuration
                #     In Proceedings of the conference on Learning and Intelligent
                #     Optimization(LION 5)
                get_one_exchange_neighbourhood(inc, seed=self._rng.randint(low=0, high=100000))
            )
            local_search_steps[i] += 1

        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq: list[list[Configuration]] = [[] for _ in range(num_candidates)]

        num_iters = 0
        while np.any(active):
            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_candidates
            # Used to request a new neighborhood for the candidates of the i-th local search
            new_neighborhood = [False] * num_candidates

            # gather all neighbors
            neighbors = []
            for i, neighborhood_iterator in enumerate(neighborhood_iterators):
                if active[i]:
                    neighbors_for_i = []
                    for j in range(obtain_n[i]):
                        try:
                            n = next(neighborhood_iterator)
                            neighbors_generated[i] += 1
                            neighbors_for_i.append(n)
                        except ValueError as e:
                            # `neighborhood_iterator` raises `ValueError` with some probability when it reaches
                            # an invalid configuration.
                            logger.debug(e)
                            new_neighborhood[i] = True
                        except StopIteration:
                            new_neighborhood[i] = True
                            break
                    obtain_n[i] = len(neighbors_for_i)
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self._acquisition_function(neighbors)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = np.asarray([acq_val])

                # Comparing the acquisition function of the neighbors with the acquisition value of the candidate
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_candidates):
                    if not active[i]:
                        continue

                    # And for each local search we know how many neighbors we obtained
                    for j in range(obtain_n[i]):
                        # The next line is only true if there was an improvement and we basically need to iterate to
                        # the i+1-th local search
                        if improved[i]:
                            acq_index += 1
                        else:
                            neighbors_looked_at[i] += 1

                            # Found a better configuration
                            if acq_val[acq_index] > acq_val_candidates[i]:
                                is_valid = False
                                try:
                                    neighbors[acq_index].is_valid_configuration()
                                    is_valid = True
                                except (ValueError, ForbiddenValueError) as e:
                                    logger.debug("Local search %d: %s", i, e)

                                if is_valid:
                                    # We comment this as it just spams the log
                                    # logger.debug(
                                    #     "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    #     i,
                                    #     neighbors_looked_at[i],
                                    # )
                                    candidates[i] = neighbors[acq_index]
                                    acq_val_candidates[i] = acq_val[acq_index]
                                    new_neighborhood[i] = True
                                    improved[i] = True
                                    local_search_steps[i] += 1
                                    neighbors_w_equal_acq[i] = []
                                    obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_candidates[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_candidates):
                if not active[i]:
                    continue

                if obtain_n[i] == 0 or improved[i]:
                    obtain_n[i] = 2
                else:
                    obtain_n[i] = obtain_n[i] * 2
                    obtain_n[i] = min(obtain_n[i], self._vectorization_max_obtain)

                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self._n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            candidates[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self._n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        candidates[i],
                        seed=self._rng.randint(low=0, high=100000),
                    )

        logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps,
            neighbors_looked_at,
            np.mean(times),
        )

        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]
