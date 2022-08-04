from __future__ import annotations

import abc
from typing import Callable, Iterator, List, Optional, Set, Tuple, Union

import copy
import itertools
import logging
import time

import numpy as np

from smac.acquisition_function import AbstractAcquisitionFunction
from smac.acquisition_optimizer import AbstractAcquisitionOptimizer
from smac.chooser.random_chooser import ChooserNoCoolDown, RandomChooser
from smac.configspace import (
    Configuration,
    ConfigurationSpace,
    ForbiddenValueError,
    convert_configurations_to_array,
    get_one_exchange_neighbourhood,
)
from smac.runhistory.runhistory import RunHistory
from smac.utils.stats import Stats


class LocalSearch(AbstractAcquisitionOptimizer):
    """Implementation of SMAC's local search.

    Parameters
    ----------
    acquisition_function : ~smac.acquisition.AbstractAcquisitionFunction
    configspace : ~smac.configspace.ConfigurationSpace
    rng : np.random.RandomState or int, optional
    max_steps: int
        Maximum number of iterations that the local search will perform
    n_steps_plateau_walk: int
        number of steps during a plateau walk before local search terminates
    vectorization_min_obtain : int
        Minimal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC
    vectorization_max_obtain : int
        Maximal number of neighbors to obtain at once for each local search for vectorized calls. Can be tuned to
        reduce the overhead of SMAC
    """

    def __init__(
        self,
        configspace: ConfigurationSpace,
        acquisition_function: AbstractAcquisitionFunction | None = None,
        max_steps: Optional[int] = None,
        n_steps_plateau_walk: int = 10,
        vectorization_min_obtain: int = 2,
        vectorization_max_obtain: int = 64,
        challengers: int = 5000,
        seed: int = 0,
    ):
        super().__init__(configspace, acquisition_function, challengers=challengers, seed=seed)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk
        self.vectorization_min_obtain = vectorization_min_obtain
        self.vectorization_max_obtain = vectorization_max_obtain

    def _maximize(
        self,
        previous_configs: List[Configuration],
        num_points: int,
        additional_start_points: Optional[List[Tuple[float, Configuration]]] = None,
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits if either the max number of
        steps is reached or no neighbor with an higher improvement was found.

        Parameters
        ----------
        previous_configs: List[Configuration]
            Previously evaluated configurations.
        response_values: np.ndarray | List[float]
            response values of the configurations
        num_points: int
            number of points to be sampled
        additional_start_points : Optional[List[Tuple[float, Configuration]]]
            Additional start point

        Returns
        -------
        List
        """
        init_points = self._get_initial_points(num_points, previous_configs, additional_start_points)
        configs_acq = self._do_search(init_points)

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        for _, inc in configs_acq:
            inc.origin = "Local Search"

        return configs_acq

    def _get_initial_points(
        self,
        num_points: int,
        previous_configs: List[Configuration],
        additional_start_points: Optional[List[Tuple[float, Configuration]]],
    ) -> List[Configuration]:

        if len(previous_configs) == 0:
            init_points = self.configspace.sample_configuration(size=num_points)
        else:
            init_points = self._get_init_points_from_previous_configs(
                num_points, previous_configs, additional_start_points
            )
        return init_points

    def _get_init_points_from_previous_configs(
        self,
        num_points: int,
        previous_configs: List[Configuration],
        additional_start_points: Optional[List[Tuple[float, Configuration]]],
    ) -> List[Configuration]:
        """
        A function that generates a set of initial points from the previous configurations and additional points (if
        applicable). The idea is to decouple runhistory from the local search model and replace it with a more genreal
        form (List[Configuration]).

        Parameters
        ----------
        num_points: int
            Number of initial points to be generated
        previous_configs: List[Configuration]
            Previous configuration from runhistory
        additional_start_points: Optional[List[Tuple[float, Configuration]]]
            if we want to specify another set of points as initial points

        Returns
        -------
        init_points: List[Configuration]
            a set of initial points
        """
        assert self.acquisition_function is not None

        # configurations with the highest previous EI
        configs_previous_runs_sorted = self._sort_configs_by_acq_value(previous_configs)
        configs_previous_runs_sorted = [conf[1] for conf in configs_previous_runs_sorted[:num_points]]

        # configurations with the lowest predictive cost, check for None to make unit tests work
        if self.acquisition_function.model is not None:
            conf_array = convert_configurations_to_array(previous_configs)
            costs = self.acquisition_function.model.predict_marginalized_over_instances(conf_array)[0]
            assert len(conf_array) == len(costs), (conf_array.shape, costs.shape)

            # In case of the predictive model returning the prediction for more than one objective per configuration
            # (for example multi-objective or EIPS) it is not immediately clear how to sort according to the cost
            # of a configuration. Therefore, we simply follow the ParEGO approach and use a random scalarization.
            if len(costs.shape) == 2 and costs.shape[1] > 1:
                weights = np.array([self.rng.rand() for _ in range(costs.shape[1])])
                weights = weights / np.sum(weights)
                costs = costs @ weights

            # From here
            # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            random = self.rng.rand(len(costs))
            # Last column is primary sort key!
            indices = np.lexsort((random.flatten(), costs.flatten()))

            # Cannot use zip here because the indices array cannot index the
            # rand_configs list, because the second is a pure python list
            previous_configs_sorted_by_cost = [previous_configs[ind] for ind in indices][:num_points]
        else:
            previous_configs_sorted_by_cost = []

        if additional_start_points is not None:
            additional_start_points = [asp[1] for asp in additional_start_points[:num_points]]
        else:
            additional_start_points = []

        init_points = []
        init_points_as_set = set()  # type: Set[Configuration]
        for cand in itertools.chain(
            configs_previous_runs_sorted,
            previous_configs_sorted_by_cost,
            additional_start_points,
        ):
            if cand not in init_points_as_set:
                init_points.append(cand)
                init_points_as_set.add(cand)
        return init_points

    def _do_search(
        self,
        start_points: List[Configuration],
    ) -> List[Tuple[float, Configuration]]:
        assert self.acquisition_function is not None

        # Gather data structure for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        candidates = start_points
        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates_ = self.acquisition_function(candidates)
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
        obtain_n = [self.vectorization_min_obtain] * num_candidates
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(candidates):
            neighborhood_iterators.append(
                get_one_exchange_neighbourhood(inc, seed=self.rng.randint(low=0, high=100000))
            )
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[] for _ in range(num_candidates)]  # type: List[List[Configuration]]

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
                            self.logger.debug(e)
                            new_neighborhood[i] = True
                        except StopIteration:
                            new_neighborhood[i] = True
                            break
                    obtain_n[i] = len(neighbors_for_i)
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self.acquisition_function(neighbors)
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
                                    self.logger.debug("Local search %d: %s", i, e)
                                if is_valid:
                                    self.logger.debug(
                                        "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                        i,
                                        neighbors_looked_at[i],
                                    )
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
                    obtain_n[i] = min(obtain_n[i], self.vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < self.n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            candidates[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self.n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        candidates[i],
                        seed=self.rng.randint(low=0, high=100000),
                    )

        self.logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps,
            neighbors_looked_at,
            np.mean(times),
        )

        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]
