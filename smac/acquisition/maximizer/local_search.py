from __future__ import annotations

from typing import Any

import itertools
import time
import warnings

import numpy as np
from ConfigSpace import Configuration, ConfigurationSpace
from ConfigSpace.exceptions import ForbiddenValueError
from ConfigSpace.hyperparameters import (
    CategoricalHyperparameter,
    OrdinalHyperparameter,
    UniformIntegerHyperparameter,
)
from joblib import Parallel, delayed

from smac.acquisition.function import AbstractAcquisitionFunction
from smac.acquisition.maximizer.abstract_acquisition_maximizer import (
    AbstractAcquisitionMaximizer,
)
from smac.utils.configspace import (
    convert_configurations_to_array,
    get_k_exchange_neighbourhood,
    get_one_exchange_neighbourhood,
)
from smac.utils.logging import get_logger

__copyright__ = "Copyright 2025, Leibniz University Hanover, Institute of AI"
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
    n_jobs : int, defaults to 1
        Number of parallel jobs to use when performing local search. If 1, the search is serial.
        If >1, multiple starting points are evaluated in parallel.
    base_sigma: float, defaults to 2
        Base standard deviation for sampling continous hyperparameters
    exchange_size : int, defaults to 1
        Number of hyperparameters to modify in each neighborhood step.

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
        n_jobs: int = 1,
        base_sigma: float = 0.2,
        exchange_size: int = 1,
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
        self._n_jobs = n_jobs
        self._base_sigma = base_sigma
        self._exchange_size = exchange_size

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
            if n_points - len(previous_configs) == 1:
                sampled_points = [self._configspace.sample_configuration()]
            else:
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

        init_points_as_set: set[Configuration] = set(
            itertools.chain(
                previous_configs_sorted_by_cost,
                additional_start_points,
            )
        )
        return list(init_points_as_set)

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

        number_of_hyperparameters = len(start_points[0].config_space.keys())
        if self._exchange_size > number_of_hyperparameters:
            warnings.warn(
                f"Requested _exchange_size={self._exchange_size} exceeds the number of "
                f"available hyperparameters ({number_of_hyperparameters}). "
                f"Setting _exchange_size to {number_of_hyperparameters}",
            )
            self._exchange_size = number_of_hyperparameters

        # Gather data structure for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]

        results = Parallel(n_jobs=self._n_jobs)(delayed(self._single_local_search)(sp) for sp in start_points)

        return results

    def _single_local_search(self, start_point: Configuration) -> tuple[float, Configuration]:
        """
        Perform a local search from a single starting configuration.

        The local search iteratively explores the k-exchange neighborhood of the
        current candidate configuration. If a neighbor has a better acquisition value,
        it becomes the new candidate. Plateau walks are used when neighbors have equal acquisition values.


        Parameters
        ----------
        start_point : Configuration
            Starting point for the search.

        Returns
        -------
        tuple[float, Configuration]
            Candidate with its acquisition function value. (acq value, candidate)
        """
        rng = np.random.RandomState(self._rng.randint(low=0, high=10000))

        candidate = start_point
        candidate_list = [candidate]
        # Compute the acquisition value of the candidate
        if self._acquisition_function is None:
            raise ValueError("Acquisition function must be set before running local search.")

        acq_val_candidate = self._acquisition_function(candidate_list)[0][0]

        # Set up additional variables required to do vectorized local search:
        # whether the local search is still running
        active = True
        # number of plateau walks of the local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = 0
        # tracking the number of steps for logging purposes
        local_search_steps = 0
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = 0
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = 0
        # how many neighbors were obtained for the local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = self._vectorization_min_obtain
        # Tracking the time it takes to compute the acquisition function
        times = []

        local_search_steps += 1
        neighbors_w_equal_acq: list[Configuration] = []

        hp_names = list(candidate.config_space.keys())

        first_iteration = True

        num_iters = 1
        while active:

            # Compute standard deviation based on Regis and Shoemaker (2013)
            if self._max_steps is not None:
                sigma_t = self._base_sigma * (1 - np.log(num_iters + 1) / np.log(self._max_steps + 1))
            else:
                sigma_t = self._base_sigma

            # Set up the neighborhood generator
            if first_iteration:
                if self._exchange_size == 1:
                    neighborhood_iterator = get_one_exchange_neighbourhood(
                        candidate,
                        seed=rng.randint(low=0, high=100000),
                        stdev=sigma_t,
                    )
                elif self._exchange_size > 1:
                    neighborhood_iterator = get_k_exchange_neighbourhood(
                        candidate,
                        seed=rng.randint(low=0, high=100000),
                        stdev=sigma_t,
                        exchange_size=self._exchange_size,
                    )
                first_iteration = False

            # If the maximum number of steps is reached, stop the local search
            if num_iters is not None and num_iters == self._max_steps:
                break

            num_iters += 1

            hp_names = list(candidate.config_space.keys())

            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = False
            # Used to request a new neighborhood for the candidates of the i-th local search
            new_neighborhood = False
            exhausted_hp = set()
            regen_count = {hp: 0 for hp in candidate.config_space}

            # gather all neighbors
            neighbors = []

            for _ in range(obtain_n):
                try:
                    n = next(neighborhood_iterator)

                    # Lists containing each hyperparameter that was changed by the neighborhood_iterator
                    changed_hp_idx = (n.get_array() != candidate.get_array()).nonzero()[0]
                    changed_hp_names = [hp_names[i] for i in changed_hp_idx]

                    for hp_name in changed_hp_names:
                        regen_count[hp_name] = regen_count.get(hp_name, 0) + 1
                        node = candidate.config_space[hp_name]

                        # number of possible values for this hypeparameter
                        n_values = (
                            len(node.choices)
                            if isinstance(node, CategoricalHyperparameter)
                            else node.size
                            if isinstance(node, UniformIntegerHyperparameter)
                            else len(node.sequence)
                            if isinstance(node, OrdinalHyperparameter)
                            else np.inf
                        )

                        # Stop adding neighbors that adjust this hyperparameter,
                        # as all possible configurations were probably tried already
                        if n_values <= 1.5 * regen_count[hp_name]:
                            exhausted_hp.add(hp_name)

                    if all(hp in exhausted_hp for hp in changed_hp_names):
                        continue

                    neighbors_generated += 1
                    neighbors.append(n)
                except ValueError as e:
                    # `neighborhood_iterator` raises `ValueError` with some probability when it reaches
                    # an invalid configuration.
                    logger.debug(e)
                    new_neighborhood = True
                except StopIteration:
                    new_neighborhood = True
                    break
            obtain_n = len(neighbors)
            if len(neighbors) > 0:
                start_time = time.time()
                acq_val = self._acquisition_function(neighbors)
                times.append(time.time() - start_time)

                for idx, neighbor in enumerate(neighbors):
                    neighbors_looked_at += 1
                    val = acq_val[idx][0]
                    if val > acq_val_candidate:
                        try:
                            neighbor.check_valid_configuration()
                            candidate = neighbor
                            acq_val_candidate = val
                            new_neighborhood = True
                            improved = True
                            local_search_steps += 1
                            neighbors_w_equal_acq = []
                            obtain_n = 1
                            break
                        except (ValueError, ForbiddenValueError) as e:
                            logger.debug("Local search: %s", e)
                    elif val == acq_val_candidate:
                        neighbors_w_equal_acq.append(neighbor)
            if obtain_n == 0 or improved:
                obtain_n = 2
            else:
                obtain_n = min(obtain_n * 2, self._vectorization_max_obtain)
            if new_neighborhood:
                if not improved and n_no_plateau_walk < self._n_steps_plateau_walk:
                    if len(neighbors_w_equal_acq) > 0:
                        candidate = neighbors_w_equal_acq[0]
                        neighbors_w_equal_acq = []
                    n_no_plateau_walk += 1
                if n_no_plateau_walk >= self._n_steps_plateau_walk:
                    active = False
                    break

                if self._exchange_size == 1:
                    neighborhood_iterator = get_one_exchange_neighbourhood(
                        candidate,
                        seed=rng.randint(low=0, high=100000),
                        stdev=sigma_t,
                    )
                elif self._exchange_size > 1:
                    neighborhood_iterator = get_k_exchange_neighbourhood(
                        candidate,
                        seed=rng.randint(low=0, high=100000),
                        stdev=sigma_t,
                        exchange_size=self._exchange_size,
                    )

        logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps,
            neighbors_looked_at,
            np.mean(times),
        )

        return acq_val_candidate, candidate
