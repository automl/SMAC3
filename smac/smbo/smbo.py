import itertools
import logging
import numpy as np
import os
import random
import sys
import time

import ConfigSpace.util

from smac.smbo.acquisition import AbstractAcquisitionFunction
from smac.smbo.base_solver import BaseSolver
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.intensification.intensification import Intensifier
from smac.smbo import pSMAC
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.stats.stats import Stats
from smac.initial_design.initial_design import InitialDesign
from smac.scenario.scenario import Scenario

from smac.epm.rfr_imputator import RFRImputator

__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
#__maintainer__ = "???"
#__email__ = "???"
__version__ = "0.0.1"


class SMBO(BaseSolver):

    def __init__(self, 
                 scenario:Scenario,
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
                 rng:np.random.RandomState):
        '''
        Interface that contains the main Bayesian optimization loop

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
            Object that implements the AbstractRunHistory2EPM to convert runhistory data into EPM data
        intensifier: Intensifier
            intensification of new challengers against incumbent configuration (probably with some kind of racing on the instances)
        aggregate_func: callable
            how to aggregate the runs in the runhistory to get the performance of a configuration
        num_run: int
            id of this run (used for pSMAC)
        model: RandomForestWithInstances
            empirical performance model (right now, we support only RandomForestWithInstances)
        acq_optimizer: LocalSearch
            optimizer on acquisition function (right now, we support only a local search)
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill criterion for acq_optimizer)
        rng: np.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger("SMBO")
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
        '''
        Runs the Bayesian optimization loop 

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        '''
        self.stats.start_timing()
        self.incumbent = self.initial_design.run()

        # Main BO loop
        iteration = 1
        while True:
            if self.scenario.shared_model:
                pSMAC.read(run_history=self.runhistory,
                           output_directory=self.scenario.output_dir,
                           configuration_space=self.config_space,
                           logger=self.logger)

            start_time = time.time()
            X, Y = self.rh2EPM.transform(self.runhistory)

            self.logger.debug("Search for next configuration")
            # get all found configurations sorted according to acq
            challengers = self.choose_next(X, Y)

            time_spend = time.time() - start_time
            logging.debug(
                "Time spend to choose next configurations: %.2f sec" % (time_spend))

            self.logger.debug("Intensify")

            self.incumbent, inc_perf = self.intensifier.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                aggregate_func=self.aggregate_func,
                time_bound=max(0.01, time_spend))

            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir,
                            num_run=self.num_run)

            iteration += 1

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self, X, Y, num_interleaved_random=1010,
                    num_configurations_by_random_search_sorted=1000,
                    num_configurations_by_local_search=10):
        """Choose next candidate solution with Bayesian optimization.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.

        Returns
        -------
        list
            List of 2020 suggested configurations to evaluate.
        """
        self.model.train(X, Y)

        if self.runhistory.empty():
            incumbent_value = 0.0
        elif self.incumbent is None:
            # TODO try to calculate an incumbent from the runhistory!
            incumbent_value = 0.0
        else:
            incumbent_value = self.runhistory.get_cost(self.incumbent)

        self.acquisition_func.update(model=self.model, eta=incumbent_value)

        # Remove dummy acquisition function value
        next_configs_by_random_search = [x[1] for x in
                                         self._get_next_by_random_search(num_points=num_interleaved_random)]

        # Get configurations sorted by EI
        next_configs_by_random_search_sorted = \
            self._get_next_by_random_search(
                num_configurations_by_random_search_sorted, _sorted=True)
        next_configs_by_local_search = \
            self._get_next_by_local_search(num_configurations_by_local_search)

        next_configs_by_acq_value = next_configs_by_random_search_sorted + \
            next_configs_by_local_search
        next_configs_by_acq_value.sort(reverse=True, key=lambda x: x[0])
        self.logger.debug(
            "First 10 acq func values of selected configurations: %s" %
            (str([_[0] for _ in next_configs_by_acq_value[:10]])))
        next_configs_by_acq_value = [_[1] for _ in next_configs_by_acq_value]

        challengers = list(itertools.chain(*zip(next_configs_by_acq_value,
                                                next_configs_by_random_search)))
        return challengers

    def _get_next_by_random_search(self, num_points=1000, _sorted=False):
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

        rand_configs = self.config_space.sample_configuration(size=num_points)
        if _sorted:
            imputed_rand_configs = map(ConfigSpace.util.impute_inactive_values,
                                       rand_configs)
            imputed_rand_configs = [x.get_array()
                                    for x in imputed_rand_configs]
            imputed_rand_configs = np.array(imputed_rand_configs,
                                            dtype=np.float64)
            acq_values = self.acquisition_func(imputed_rand_configs)
            # From here
            # http://stackoverflow.com/questions/20197990/how-to-make-argsort-result-to-be-random-between-equal-values
            random = self.rng.rand(len(acq_values))
            # Last column is primary sort key!
            indices = np.lexsort((random.flatten(), acq_values.flatten()))

            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search (sorted)'

            # Cannot use zip here because the indices array cannot index the
            # rand_configs list, because the second is a pure python list
            return [(acq_values[ind][0], rand_configs[ind])
                    for ind in indices[::-1]]
        else:
            for i in range(len(rand_configs)):
                rand_configs[i].origin = 'Random Search'
            return [(0, rand_configs[i]) for i in range(len(rand_configs))]

    def _get_next_by_local_search(self, num_points=10):
        """Get candidate solutions via local search.

        In case acquisition function values tie, these will be broken randomly.

        Parameters
        ----------
        num_points : int, optional (default=10)
            Number of local searches and returned values.

        Returns
        -------
        list : (acquisition value, Candidate solutions),
               ordered by their acquisition function value
        """
        configs_acq = []

        # Start N local search from different random start points
        for i in range(num_points):
            if i == 0 and self.incumbent is not None:
                start_point = self.incumbent
            else:
                start_point = self.config_space.sample_configuration()

            configuration, acq_val = self.acq_optimizer.maximize(start_point)

            configuration.origin = 'Local Search'
            configs_acq.append((acq_val[0][0], configuration))

        # shuffle for random tie-break
        random.shuffle(configs_acq, self.rng.rand)

        # sort according to acq value
        # and return n best configurations
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq
