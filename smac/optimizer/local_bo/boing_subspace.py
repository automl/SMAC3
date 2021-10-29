import typing
import itertools
import time
import numpy as np

from ConfigSpace import ConfigurationSpace

from smac.configspace import Configuration
from smac.configspace import get_one_exchange_neighbourhood
from smac.optimizer.local_bo.abstract_subspace import AbstractSubspace
from smac.epm.base_epm import AbstractEPM
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI
from smac.epm.partial_sparse_gaussian_process import PartialSparseGaussianProcess


class BOinGSubspace(AbstractSubspace):
    def __init__(self,
                 config_space: ConfigurationSpace,
                 bounds: typing.List[typing.Tuple[float, float]],
                 hps_types: typing.List[int],
                 bounds_ss_cont: typing.Optional[np.ndarray] = None,
                 bounds_ss_cat: typing.Optional[typing.List[typing.Tuple]] = None,
                 model_local: AbstractEPM = PartialSparseGaussianProcess,
                 model_local_kwargs: typing.Optional[typing.Dict] = None,
                 acq_func_local: AbstractAcquisitionFunction = EI,
                 acq_func_local_kwargs: typing.Optional[typing.Dict] = None,
                 rng: typing.Optional[np.random.RandomState] = None,
                 initial_data: typing.Optional[typing.Tuple[np.ndarray, np.ndarray]] = None,
                 activate_dims: typing.Optional[np.ndarray] = None,
                 incumbent_array: typing.Optional[np.ndarray] = None,
                 ):
        super(BOinGSubspace, self).__init__(config_space=config_space,
                                            bounds=bounds,
                                            hps_types=hps_types,
                                            bounds_ss_cont=bounds_ss_cont,
                                            bounds_ss_cat=bounds_ss_cat,
                                            model_local=model_local,
                                            model_local_kwargs=model_local_kwargs,
                                            acq_func_local=acq_func_local,
                                            acq_func_local_kwargs=acq_func_local_kwargs,
                                            rng=rng,
                                            initial_data=initial_data,
                                            activate_dims=activate_dims,
                                            incumbent_array=incumbent_array)
        self.config_origin = "BOinG"
        if isinstance(self.model, PartialSparseGaussianProcess):
            num_inducing_points = min(max(min(2 * len(self.activate_dims_cont), 10), self.model_x.shape[0] // 20), 50)
            self.model.update_attribute(num_inducing_points=num_inducing_points)

    def _generate_challengers(self):
        """
        generate new challengers list for this subspace, this optimizer is similar to
        smac.optimizer.ei_optimization.LocalAndSortedRandomSearch
        """
        self.model.train(self.model_x, self.model_y)
        self.update_model(predict_x_best=True, update_incumbent_array=True)

        num_random_samples = 10000
        num_init_points = {
            1: 10,
            2: 10,
            3: 10,
            4: 10,
            5: 10,
            6: 10,
            7: 8,
            8: 6,
        }.get(len(self.cs_local.get_hyperparameters()), 5)

        vectorization_min_obtain = 2
        vectorization_max_obtain = 64
        n_steps_plateau_walk = 5

        samples_random = self.cs_local.sample_configuration(size=num_random_samples)
        acq_values_random = self.acquisition_function(samples_random)

        random = self.rng.rand(len(acq_values_random))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values_random.flatten()))
        candidates_random = [(acq_values_random[ind], samples_random[ind]) for ind in indices]

        init_points_local = self._get_init_points(num_init_points=num_init_points,
                                                  additional_start_points=candidates_random)
        candidates_local = self._do_search(start_points=init_points_local,
                                           vectorization_min_obtain=vectorization_min_obtain,
                                           vectorization_max_obtain=vectorization_max_obtain,
                                           n_steps_plateau_walk=n_steps_plateau_walk
                                           )

        next_configs_by_acq_value = candidates_random + candidates_local

        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])

        return next_configs_by_acq_value

    def _get_init_points(self,
                         num_init_points: int,
                         additional_start_points: typing.Optional[typing.List[typing.Tuple[float, Configuration]]]):
        if len(self.ss_x) == 0:
            init_points = self.cs_local.sample_configuration(size=num_init_points)
        else:
            stored_configs = [Configuration(configuration_space=self.cs_local, vector=ss_x) for ss_x in self.ss_x]
            acq_values_previous = self.acquisition_function(stored_configs)
            random = self.rng.rand(len(acq_values_previous))
            # Last column is primary sort key!
            indices = np.lexsort((random.flatten(), acq_values_previous.flatten()))
            configs_previous_runs_sorted = [stored_configs[ind] for ind in indices[::-1]][:num_init_points]

            costs, _ = self.model.predict_marginalized_over_instances(self.ss_x)
            assert len(self.ss_x) == len(costs), (self.ss_x.shape, costs.shape)

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
            configs_previous_runs_sorted_by_cost = [stored_configs[ind] for ind in indices][:num_init_points]

            if additional_start_points is not None:
                additional_start_points = [asp[1] for asp in additional_start_points[:num_init_points]]
            else:
                additional_start_points = []

            init_points = []
            init_points_as_set = set()  # type: Set[Configuration]
            for cand in itertools.chain(
                    configs_previous_runs_sorted,
                    configs_previous_runs_sorted_by_cost,
                    additional_start_points,
            ):
                if cand not in init_points_as_set:
                    init_points.append(cand)
                    init_points_as_set.add(cand)
        return init_points

    def _do_search(
            self,
            start_points: typing.List[Configuration],
            vectorization_min_obtain: int = 2,
            vectorization_max_obtain: int = 64,
            n_steps_plateau_walk: int = 10,
    ) -> typing.List[typing.Tuple[float, Configuration]]:
        # Gather data strucuture for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        candidates = start_points
        # Compute the acquisition value of the candidates
        num_candidates = len(candidates)
        acq_val_candidates = self.acquisition_function(candidates)
        if num_candidates == 1:
            acq_val_candidates = [acq_val_candidates[0][0]]
        else:
            acq_val_candidates = [a[0] for a in acq_val_candidates]

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
        obtain_n = [vectorization_min_obtain] * num_candidates
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(candidates):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_candidates  # type: List[List[Configuration]]

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
                        except StopIteration:
                            obtain_n[i] = len(neighbors_for_i)
                            new_neighborhood[i] = True
                            break
                    neighbors.extend(neighbors_for_i)

            if len(neighbors) != 0:
                start_time = time.time()
                acq_val = self.acquisition_function(neighbors)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

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
                    obtain_n[i] = min(obtain_n[i], vectorization_max_obtain)
                if new_neighborhood[i]:
                    if not improved[i] and n_no_plateau_walk[i] < n_steps_plateau_walk:
                        if len(neighbors_w_equal_acq[i]) != 0:
                            candidates[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        candidates[i], seed=self.rng.randint(low=0, high=100000),
                    )
        return [(a, i) for a, i in zip(acq_val_candidates, candidates)]
