import itertools
import logging
import numpy as np
import random
import time
import typing
import math


from smac.optimizer.acquisition import AbstractAcquisitionFunction
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.optimizer.local_search import LocalSearch
from smac.intensification.intensification import Intensifier
from smac.optimizer import pSMAC
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.stats.stats import Stats
from smac.initial_design.initial_design import InitialDesign
from smac.scenario.scenario import Scenario
from smac.configspace import Configuration, convert_configurations_to_array
from smac.tae.execute_ta_run import FirstRunCrashedException


__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"


class SMBO(object):

    """Interface that contains the main Bayesian optimization loop

    Attributes
    ----------
    logger
    incumbent
    scenario
    config_space
    stats
    initial_design
    runhistory
    rh2EPM
    intensifier
    aggregate_func
    num_run
    model
    acq_optimizer
    acquisition_func
    rng
    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: Intensifier,
                 aggregate_func: callable,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: LocalSearch,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState):
        """Constructor

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        stats: Stats
            statistics object with configuration budgets
        initial_design: InitialDesign
            initial sampling design
        runhistory: RunHistory
            runhistory with all runs so far
        runhistory2epm : AbstractRunHistory2EPM
            Object that implements the AbstractRunHistory2EPM to convert runhistory
            data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration
            (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a
             configuration
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only
            RandomForestWithInstances)
        acq_optimizer: LocalSearch
            optimizer on acquisition function (right now, we support only a local
            search)
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill
            criterion for acq_optimizer)
        rng: np.random.RandomState
            Random number generator
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = None

        self.scenario = scenario
        self.config_space = scenario.cs
        self.stats = stats
        self.initial_design = initial_design
        self.runhistory = runhistory
        self.rh2EPM = runhistory2epm
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.model = model
        self.acq_optimizer = acq_optimizer
        self.acquisition_func = acquisition_func
        self.rng = rng

    def run(self):
        """Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        """
        self.stats.start_timing()
        try:
            self.incumbent = self.initial_design.run()
        except FirstRunCrashedException as err:
            if self.scenario.abort_on_first_run_crash:
                raise

        # Main BO loop
        iteration = 1
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_dirs=self.scenario.input_psmac_dirs,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            time_spent = time.time() - start_time
            time_left = self._get_timebound_for_intensification(time_spent)

            self.logger.debug("Intensify")

            self.incumbent, inc_perf = self.intensifier.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
                time_bound=max(self.intensifier._min_time, time_left))

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir)

            iteration += 1

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self, X: np.ndarray, Y: np.ndarray,
                    num_configurations_by_random_search_sorted: int=1000,
                    num_configurations_by_local_search: int=None,
                    incumbent_value: float=None):
        """Choose next candidate solution with Bayesian optimization.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        num_configurations_by_random_search_sorted: int
            Number of configurations optimized by random search
        num_configurations_by_local_search: int
            Number of configurations optimized with local search
            if None, we use min(10, 1 + 0.5 x the number of configurations on
            exp average in intensify calls)
        incumbent_value: float
            Cost value of incumbent configuration
            (required for acquisition function);
            if not given, it will be inferred from runhistory;
            if not given and runhistory is empty,
            it will raise a ValueError

        Returns
        -------
        list
            List of 2020 suggested configurations to evaluate.
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return [x[1] for x in self._get_next_by_random_search(num_points=1)]

        self.model.train(X, Y)

        if incumbent_value is None:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent_value = self.runhistory.get_cost(self.incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = \
            self._get_next_by_random_search(
                num_configurations_by_random_search_sorted, _sorted=True)

        if num_configurations_by_local_search is None:
            if self.stats._ema_n_configs_per_intensifiy > 0:
                num_configurations_by_local_search = \
                    min(10, math.ceil(0.5 *
                                      self.stats._ema_n_configs_per_intensifiy)
                        + 1)
            else:
                num_configurations_by_local_search = 10

        if self.runhistory.empty():
            init_sls_points = self.config_space.sample_configuration(
                size=num_configurations_by_local_search)
        else:
            # initiate local search with best configurations from previous runs
            configs_previous_runs = self.runhistory.get_all_configs()
            configs_previous_runs_sorted = self._sort_configs_by_acq_value(
                configs_previous_runs)
            num_configs_local_search = \
                min(len(configs_previous_runs_sorted),
                    num_configurations_by_local_search)
            init_sls_points = list(map(lambda x: x[1],
                                       configs_previous_runs_sorted[:num_configs_local_search]))

        next_configs_by_local_search = \
            self._get_next_by_local_search(init_sls_points)

        # Having the configurations from random search, sorted by their
        # acquisition function value is important for the first few iterations
        # of SMAC. As long as the random forest predicts constant value, we
        # want to use only random configurations. Having them at the begging of
        # the list ensures this (even after adding the configurations by local
        # search, and then sorting them)
        next_configs_by_acq_value = next_configs_by_random_search_sorted + \
            next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func (origin) values of selected configurations: %s" %
            (str([[_[0], _[1].origin] for _ in next_configs_by_acq_value[:10]])))
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = ChallengerList(next_configs_by_acq_value,
                                     self.config_space)
        return challengers

    def _get_next_by_random_search(self, num_points: int=1000,
                                   _sorted: bool=False):
        """Get candidate solutions via local search.

        Parameters
        ----------
        num_points : int, optional (default=10)
            Number of local searches and returned values.

        _sorted : bool, optional (default=True)
            Whether to sort the candidate solutions by acquisition function
            value.

        Returns
        -------
        list : (acquisition value, Candidate solutions)
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

    def _get_next_by_local_search(self,
                                  init_points=typing.List[Configuration]):
        """Get candidate solutions via local search.

        In case acquisition function values tie, these will be broken randomly.

        Parameters
        ----------
        init_points : typing.List[Configuration]
            initial starting configurations for local search

        Returns
        -------
        list : (acquisition value, Candidate solutions),
               ordered by their acquisition function value
        """
        configs_acq = []

        # Start N local search from different random start points
        for start_point in init_points:
            configuration, acq_val = self.acq_optimizer.maximize(start_point)

            configuration.origin = "Local Search"
            configs_acq.append((acq_val[0], configuration))

        # shuffle for random tie-break
        random.shuffle(configs_acq, self.rng.rand)

        # sort according to acq value
        # and return n best configurations
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq

    def _sort_configs_by_acq_value(self, configs):
        """Sort the given configurations by acquisition value

        Parameters
        ----------
        configs : list(Configuration)

        Returns
        -------
        list: (acquisition value, Candidate solutions),
                ordered by their acquisition function value

        """

        config_array = convert_configurations_to_array(configs)
        acq_values = self.acquisition_func(config_array)

        # From here
        # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
        random = self.rng.rand(len(acq_values))
        # Last column is primary sort key!
        indices = np.lexsort((random.flatten(), acq_values.flatten()))

        # Cannot use zip here because the indices array cannot index the
        # rand_configs list, because the second is a pure python list
        return [(acq_values[ind][0], configs[ind]) for ind in indices[::-1]]

    def _get_timebound_for_intensification(self, time_spent):
        """Calculate time left for intensify from the time spent on
        choosing challengers using the fraction of time intended for
        intensification (which is specified in
        scenario.intensification_percentage).

        Parameters
        ----------
        time_spent : float

        Returns
        -------
        time_left : float
        """
        frac_intensify = self.scenario.intensification_percentage
        if frac_intensify <= 0 or frac_intensify >= 1:
            raise ValueError("The value for intensification_percentage-"
                             "option must lie in (0,1), instead: %.2f" %
                             (frac_intensify))
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug("Total time: %.4f, time spent on choosing next "
                          "configurations: %.4f (%.2f), time left for "
                          "intensification: %.4f (%.2f)" %
                          (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left


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

    def __init__(self, challengers, configuration_space):
        self.challengers = challengers
        self.configuration_space = configuration_space
        self._index = 0
        self._next_is_random = False

    def __iter__(self):
        return self

    def __next__(self):
        if self._index == len(self.challengers) and not self._next_is_random:
            raise StopIteration
        elif self._next_is_random:
            self._next_is_random = False
            config = self.configuration_space.sample_configuration()
            config.origin = 'Random Search'
            return config
        else:
            self._next_is_random = True
            config = self.challengers[self._index]
            self._index += 1
            return config
