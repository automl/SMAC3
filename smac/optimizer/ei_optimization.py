import abc
import logging
import time
import numpy as np

from typing import Iterable, List, Union, Tuple, Optional

from smac.configspace import get_one_exchange_neighbourhood, Configuration, ConfigurationSpace
from smac.runhistory.runhistory import RunHistory
from smac.stats.stats import Stats
from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown
from smac.utils.constants import MAXINT

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Aaron Klein"
__email__ = "kleinaa@cs.uni-freiburg.de"
__version__ = "0.0.1"


class AcquisitionFunctionMaximizer(object, metaclass=abc.ABCMeta):
    """Abstract class for acquisition maximization.

    In order to use this class it has to be subclassed and the method
    ``_maximize`` must be implemented.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None
    ):
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__
        )
        self.acquisition_function = acquisition_function
        self.config_space = config_space

        if rng is None:
            self.logger.debug('no rng given, using default seed of 1')
            self.rng = np.random.RandomState(seed=1)
        else:
            self.rng = rng


    def maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consisting of :class:`smac.configspace.Configuration`.
        """
        return [t[1] for t in self._maximize(runhistory, stats, num_points, **kwargs)]

    @abc.abstractmethod
    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        """Implements acquisition function maximization.

        In contrast to ``maximize``, this method returns an iterable of tuples,
        consisting of the acquisition function value and the configuration. This
        allows to plug together different acquisition function maximizers.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        **kwargs

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.Configuration`).
        """
        raise NotImplementedError()

    def _sort_configs_by_acq_value(
            self,
            configs: List[Configuration]
    ) -> List[Tuple[float, Configuration]]:
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value
        """

        acq_values = self.acquisition_function(configs)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]


class LocalSearch(AcquisitionFunctionMaximizer):

    """Implementation of SMAC's local search.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

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
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            vectorization_min_obtain: int = 2,
            vectorization_max_obtain: int = 64,
    ):
        super().__init__(acquisition_function, config_space, rng)
        self.max_steps = max_steps
        self.n_steps_plateau_walk = n_steps_plateau_walk
        self.vectorization_min_obtain = vectorization_min_obtain
        self.vectorization_max_obtain = vectorization_max_obtain

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Starts a local search from the given startpoint and quits
        if either the max number of steps is reached or no neighbor
        with an higher improvement was found.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        ***kwargs:
            Additional parameters that will be passed to the
            acquisition function

        Returns
        -------
        incumbent: np.array(1, D)
            The best found configuration
        acq_val_incumbent: np.array(1,1)
            The acquisition value of the incumbent

        """

        init_points = self._get_initial_points(num_points, runhistory)
        configs_acq = self._do_search(init_points)

        # shuffle for random tie-break
        self.rng.shuffle(configs_acq)

        # sort according to acq value
        configs_acq.sort(reverse=True, key=lambda x: x[0])
        for _, inc in configs_acq:
            inc.origin = 'Local Search'

        return configs_acq

    def _get_initial_points(self, num_points, runhistory):

        if runhistory.empty():
            init_points = self.config_space.sample_configuration(
                size=num_points)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = runhistory.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = int(min(
                len(configs_previous_runs_sorted),
                num_points)
            )
            init_points = list(
                map(lambda x: x[1],
                    configs_previous_runs_sorted[:num_configs_local_search])
            )

        return init_points

    def _do_search(
            self,
            start_points: List[Configuration],
            **kwargs
    ) -> List[Tuple[float, Configuration]]:

        # Gather data strucuture for starting points
        if isinstance(start_points, Configuration):
            start_points = [start_points]
        incumbents = start_points
        # Compute the acquisition value of the incumbents
        num_incumbents = len(incumbents)
        acq_val_incumbents = self.acquisition_function(incumbents, **kwargs)
        if num_incumbents == 1:
            acq_val_incumbents = [acq_val_incumbents[0][0]]
        else:
            acq_val_incumbents = [a[0] for a in acq_val_incumbents]

        # Set up additional variables required to do vectorized local search:
        # whether the i-th local search is still running
        active = [True] * num_incumbents
        # number of plateau walks of the i-th local search. Reaching the maximum number is the stopping criterion of
        # the local search.
        n_no_plateau_walk = [0] * num_incumbents
        # tracking the number of steps for logging purposes
        local_search_steps = [0] * num_incumbents
        # tracking the number of neighbors looked at for logging purposes
        neighbors_looked_at = [0] * num_incumbents
        # tracking the number of neighbors generated for logging purposse
        neighbors_generated = [0] * num_incumbents
        # how many neighbors were obtained for the i-th local search. Important to map the individual acquisition
        # function values to the correct local search run
        obtain_n = [self.vectorization_min_obtain] * num_incumbents
        # Tracking the time it takes to compute the acquisition function
        times = []

        # Set up the neighborhood generators
        neighborhood_iterators = []
        for i, inc in enumerate(incumbents):
            neighborhood_iterators.append(get_one_exchange_neighbourhood(
                inc, seed=self.rng.randint(low=0, high=100000)))
            local_search_steps[i] += 1
        # Keeping track of configurations with equal acquisition value for plateau walking
        neighbors_w_equal_acq = [[]] * num_incumbents

        num_iters = 0
        while np.any(active):

            num_iters += 1
            # Whether the i-th local search improved. When a new neighborhood is generated, this is used to determine
            # whether a step was made (improvement) or not (iterator exhausted)
            improved = [False] * num_incumbents
            # Used to request a new neighborhood for the incumbent of the i-th local search
            new_neighborhood = [False] * num_incumbents

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
                acq_val = self.acquisition_function(neighbors, **kwargs)
                end_time = time.time()
                times.append(end_time - start_time)
                if np.ndim(acq_val.shape) == 0:
                    acq_val = [acq_val]

                # Comparing the acquisition function of the neighbors with the acquisition value of the incumbent
                acq_index = 0
                # Iterating the all i local searches
                for i in range(num_incumbents):
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
                            if acq_val[acq_index] > acq_val_incumbents[i]:
                                self.logger.debug(
                                    "Local search %d: Switch to one of the neighbors (after %d configurations).",
                                    i,
                                    neighbors_looked_at[i],
                                )
                                incumbents[i] = neighbors[acq_index]
                                acq_val_incumbents[i] = acq_val[acq_index]
                                new_neighborhood[i] = True
                                improved[i] = True
                                local_search_steps[i] += 1
                                neighbors_w_equal_acq[i] = []
                                obtain_n[i] = 1
                            # Found an equally well performing configuration, keeping it for plateau walking
                            elif acq_val[acq_index] == acq_val_incumbents[i]:
                                neighbors_w_equal_acq[i].append(neighbors[acq_index])

                            acq_index += 1

            # Now we check whether we need to create new neighborhoods and whether we need to increase the number of
            # plateau walks for one of the local searches. Also disables local searches if the number of plateau walks
            # is reached (and all being switched off is the termination criterion).
            for i in range(num_incumbents):
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
                            incumbents[i] = neighbors_w_equal_acq[i][0]
                            neighbors_w_equal_acq[i] = []
                        n_no_plateau_walk[i] += 1
                    if n_no_plateau_walk[i] >= self.n_steps_plateau_walk:
                        active[i] = False
                        continue

                    neighborhood_iterators[i] = get_one_exchange_neighbourhood(
                        incumbents[i], seed=self.rng.randint(low=0, high=100000),
                    )

        self.logger.debug(
            "Local searches took %s steps and looked at %s configurations. Computing the acquisition function in "
            "vectorized for took %f seconds on average.",
            local_search_steps, neighbors_looked_at, np.mean(times),
        )

        return [(a, i) for a, i in zip(acq_val_incumbents, incumbents)]


class DiffOpt(AcquisitionFunctionMaximizer):
    """Get candidate solutions via DifferentialEvolutionSolvers.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            _sorted: bool=False,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """DifferentialEvolutionSolver

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.Configuration`).
        """


        from scipy.optimize._differentialevolution import DifferentialEvolutionSolver
        configs = []

        def func(x):
            return -self.acquisition_function([Configuration(self.config_space, vector=x)])

        ds = DifferentialEvolutionSolver(func, bounds=[[0, 1], [0, 1]], args=(),
                                    strategy='best1bin', maxiter=1000,
                                    popsize=50, tol=0.01,
                                    mutation=(0.5, 1),
                                    recombination=0.7,
                                    seed=self.rng.randint(1000), polish=True,
                                    callback=None,
                                    disp=False, init='latinhypercube', atol=0)

        rval = ds.solve()
        for pop, val in zip(ds.population, ds.population_energies):
            rc = Configuration(self.config_space, vector=pop)
            rc.origin = 'DifferentialEvolution'
            configs.append((-val, rc))

        configs.sort(key=lambda t: t[0])
        configs.reverse()
        return configs

class RandomSearch(AcquisitionFunctionMaximizer):
    """Get candidate solutions via random sampling of configurations.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional
    """

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            _sorted: bool=False,
            **kwargs
    ) -> List[Tuple[float, Configuration]]:
        """Randomly sampled configurations

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        _sorted: bool
            whether random configurations are sorted according to acquisition function
        **kwargs
            not used

        Returns
        -------
        iterable
            An iterable consistng of
            tuple(acqusition_value, :class:`smac.configspace.Configuration`).
        """

        if num_points > 1:
            rand_configs = self.config_space.sample_configuration(
                size=num_points)
        else:
            rand_configs = [self.config_space.sample_configuration(size=1)]
        if _sorted:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'
            return self._sort_configs_by_acq_value(rand_configs)
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]


class InterleavedLocalAndRandomSearch(AcquisitionFunctionMaximizer):
    """Implements SMAC's default acquisition function optimization.

    This optimizer performs local search from the previous best points
    according, to the acquisition function, uses the acquisition function to
    sort randomly sampled configurations and interleaves unsorted, randomly
    sampled configurations in between.

    Parameters
    ----------
    acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction

    config_space : ~smac.configspace.ConfigurationSpace

    rng : np.random.RandomState or int, optional

    max_steps: int
        [LocalSearch] Maximum number of steps that the local search will perform

    n_steps_plateau_walk: int
        [LocalSearch] number of steps during a plateau walk before local search terminates

    n_sls_iterations: int
        [Local Search] number of local search iterations

    """
    def __init__(
            self,
            acquisition_function: AbstractAcquisitionFunction,
            config_space: ConfigurationSpace,
            rng: Union[bool, np.random.RandomState] = None,
            max_steps: Optional[int] = None,
            n_steps_plateau_walk: int = 10,
            n_sls_iterations: int = 10

    ):
        super().__init__(acquisition_function, config_space, rng)
        self.random_search = RandomSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng
        )
        self.local_search = LocalSearch(
            acquisition_function=acquisition_function,
            config_space=config_space,
            rng=rng,
            max_steps=max_steps,
            n_steps_plateau_walk=n_steps_plateau_walk
        )
        self.n_sls_iterations = n_sls_iterations

        #=======================================================================
        # self.local_search = DiffOpt(
        #     acquisition_function=acquisition_function,
        #     config_space=config_space,
        #     rng=rng
        # )
        #=======================================================================

    def maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            random_configuration_chooser,
            **kwargs
    ) -> Iterable[Configuration]:
        """Maximize acquisition function using ``_maximize``.

        Parameters
        ----------
        runhistory: ~smac.runhistory.runhistory.RunHistory
            runhistory object
        stats: ~smac.stats.stats.Stats
            current stats object
        num_points: int
            number of points to be sampled
        random_configuration_chooser: ~smac.optimizer.random_configuration_chooser.RandomConfigurationChooser
            part of the returned ChallengerList such
            that we can interleave random configurations
            by a scheme defined by the random_configuration_chooser;
            random_configuration_chooser.next_smbo_iteration()
            is called at the end of this function
        **kwargs
            passed to acquisition function

        Returns
        -------
        Iterable[Configuration]
            to be concrete: ~smac.ei_optimization.ChallengerList
        """

        next_configs_by_local_search = self.local_search._maximize(
            runhistory, stats, self.n_sls_iterations, **kwargs
        )

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = self.random_search._maximize(
            runhistory,
            stats,
            num_points - len(next_configs_by_local_search),
            _sorted=True,
        )

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = (
            next_configs_by_random_search_sorted
            + next_configs_by_local_search
        )
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s",
            str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])
        )
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space,
                                     random_configuration_chooser)
        random_configuration_chooser.next_smbo_iteration()
        return challengers

    def _maximize(
            self,
            runhistory: RunHistory,
            stats: Stats,
            num_points: int,
            **kwargs
    ) -> Iterable[Tuple[float, Configuration]]:
        raise NotImplementedError()


class ChallengerList(object):
    """Helper class to interleave random configurations in a list of challengers.

    Provides an iterator which returns a random configuration in each second
    iteration. Reduces time necessary to generate a list of new challengers
    as one does not need to sample several hundreds of random configurations
    in each iteration which are never looked at.

    Parameters
    ----------
    challengers : list
        List of challengers (without interleaved random configurations)

    configuration_space : ConfigurationSpace
        ConfigurationSpace from which to sample new random configurations.
    """

    def __init__(self, challengers, configuration_space, random_configuration_chooser=ChooserNoCoolDown(2.0)):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._iteration = 1  # 1-based to prevent from starting with a random configuration
        self.random_configuration_chooser = random_configuration_chooser

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers):
            raise StopIteration
        else:
            if self.random_configuration_chooser.check(self._iteration):
                config = self.configuration_space.sample_configuration()
                config.origin = 'Random Search'
            else:
                config = self.challengers[self._index]
                self._index += 1
            self._iteration += 1
            return config
