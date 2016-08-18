import itertools
import logging
import numpy as np
import os
import random
import sys
import time

from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant
import ConfigSpace.util

from smac.smbo.acquisition import EI
from smac.smbo.base_solver import BaseSolver
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.smbo.intensification import Intensifier
from smac.smbo import pSMAC
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM4Cost, \
    RunHistory2EPM4LogCost
from smac.smbo.objective import average_cost, total_runtime
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.utils.io.traj_logging import TrajLogger

from smac.epm.rfr_imputator import RFRImputator

MAXINT = 2 ** 31 - 1

__author__ = "Aaron Klein, Marius Lindauer, Matthias Feurer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
#__maintainer__ = "???"
#__email__ = "???"
__version__ = "0.0.1"


def get_types(config_space, instance_features=None):
    # Extract types vector for rf from config space
    types = np.zeros(len(config_space.get_hyperparameters()),
                     dtype=np.uint)

    for i, param in enumerate(config_space.get_hyperparameters()):
        if isinstance(param, (CategoricalHyperparameter)):
            n_cats = len(param.choices)
            types[i] = n_cats

        elif isinstance(param, Constant):
            # for constants we simply set types to 0
            # which makes it a numerical parameter
            types[i] = 0
            # and we leave the bounds to be 0 for now
        elif not isinstance(param, (UniformFloatHyperparameter,
                                    UniformIntegerHyperparameter)):
            raise TypeError("Unknown hyperparameter type %s" % type(param))

    if instance_features is not None:
        types = np.hstack(
            (types, np.zeros((instance_features.shape[1]))))

    types = np.array(types, dtype=np.uint)
    return types


class SMBO(BaseSolver):

    def __init__(self, scenario, tae_runner=None, acquisition_function=None,
                 model=None, runhistory2epm=None, stats=None, rng=None):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        tae_runner: object
            object that implements the following method to call the target
            algorithm (or any other arbitrary function):
            run(self, config)
            If not set, it will be initialized with the tae.ExecuteTARunOld()
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction. Will use
            EI if not set.
        model : object
            Model that implements train() and predict(). Will use a
            RandomForest if not set.
        runhistory2epm : RunHistory2EMP
            Object that implements the AbstractRunHistory2EPM. If None,
            will use RunHistory2EPM4Cost if objective is cost or
            RunHistory2EPM4LogCost if objective is runtime.
        stats: Stats
            optional stats object
        rng: numpy.random.RandomState
            Random number generator
        '''

        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        self.runhistory = RunHistory()

        self.logger = logging.getLogger("smbo")

        if rng is None:
            self.num_run = np.random.randint(1234567980)
            self.rng = np.random.RandomState(seed=self.num_run)
        elif isinstance(rng, int):
            self.num_run = rng
            self.rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            self.num_run = rng.randint(1234567980)
            self.rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))

        self.scenario = scenario
        self.config_space = scenario.cs

        self.types = get_types(self.config_space, scenario.feature_array)
        if model is None:
            self.model = RandomForestWithInstances(self.types,
                                                   scenario.feature_array,
                                                   seed=self.rng.randint(
                                                       1234567980))
        else:
            self.model = model

        if acquisition_function is None:
            self.acquisition_func = EI(self.model)
        else:
            self.acquisition_func = acquisition_function

        self.local_search = LocalSearch(self.acquisition_func,
                                        self.config_space)
        self.incumbent = None

        if tae_runner is None:
            self.executor = ExecuteTARunOld(ta=scenario.ta,
                                            stats=self.stats,
                                            run_obj=scenario.run_obj,
                                            par_factor=scenario.par_factor)
        else:
            self.executor = tae_runner

        self.inten = Intensifier(executor=self.executor,
                                 stats=self.stats,
                                 instances=self.scenario.train_insts,
                                 cutoff=self.scenario.cutoff,
                                 deterministic=self.scenario.deterministic,
                                 run_obj_time=self.scenario.run_obj == "runtime",
                                 instance_specifics=self.scenario.instance_specific)

        num_params = len(self.config_space.get_hyperparameters())

        self.objective = average_cost
        if self.scenario.run_obj == "runtime":

            if runhistory2epm is None:
                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log10(self.scenario.cutoff)
                threshold = np.log10(self.scenario.cutoff *
                                     self.scenario.par_factor)

                imputor = RFRImputator(cs=self.config_space,
                                       rs=self.rng,
                                       cutoff=cutoff,
                                       threshold=threshold,
                                       model=self.model,
                                       change_threshold=0.01,
                                       max_iter=10)
                self.rh2EPM = RunHistory2EPM4LogCost(
                    scenario=self.scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=True,
                    impute_state=[StatusType.TIMEOUT, ],
                    imputor=imputor)
            else:
                self.rh2EPM = runhistory2epm

        elif self.scenario.run_obj == 'quality':
            if runhistory2epm is None:
                self.rh2EPM = RunHistory2EPM4Cost\
                    (scenario=self.scenario, num_params=num_params,
                     success_states=[StatusType.SUCCESS, ],
                     impute_censored_data=False, impute_state=None)
            else:
                self.rh2EPM = runhistory2epm

        else:
            raise ValueError('Unknown run objective: %s. Should be either '
                             'quality or runtime.' % self.scenario.run_obj)

        self.trajLogger = TrajLogger(
            output_dir=self.scenario.output_dir, stats=self.stats)

    def run_initial_design(self):
        '''
            runs algorithm runs for a initial design;
            default implementation: running the default configuration on
                                    a random instance-seed pair
            Side effect: adds runs to self.runhistory

            Returns
            -------
            incumbent: Configuration()
                initial incumbent configuration
        '''

        default_conf = self.config_space.get_default_configuration()
        self.incumbent = default_conf
        rand_inst_id = self.rng.randint(0, len(self.scenario.train_insts))
        # ignore instance specific values
        rand_inst = self.scenario.train_insts[rand_inst_id]

        if self.scenario.deterministic:
            initial_seed = 0
        else:
            initial_seed = random.randint(0, MAXINT)

        status, cost, runtime, additional_info = self.executor.run(
            default_conf, instance=rand_inst, cutoff=self.scenario.cutoff,
            seed=initial_seed,
            instance_specific=self.scenario.instance_specific.get(rand_inst, "0"))

        if status in [StatusType.CRASHED or StatusType.ABORT]:
            self.logger.info("First run crashed -- Abort")
            sys.exit(42)

        self.runhistory.add(config=default_conf, cost=cost, time=runtime,
                            status=status,
                            instance_id=rand_inst,
                            seed=initial_seed,
                            additional_info=additional_info)
        defaul_inst_seeds = set(
            self.runhistory.get_runs_for_config(default_conf))
        default_perf = self.objective(default_conf, self.runhistory,
                                      defaul_inst_seeds)
        self.runhistory.update_cost(default_conf, default_perf)

        self.stats.inc_changed += 1  # first incumbent
        return default_conf

    def run(self, max_iters=10):
        '''
        Runs the Bayesian optimization loop for max_iters iterations

        Parameters
        ----------
        max_iters: int
            The maximum number of iterations

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        '''
        self.stats.start_timing()

        #self.runhistory = RunHisory()

        self.incumbent = self.run_initial_design()

        self.trajLogger.add_entry(train_perf=999999999,
                                  incumbent_id=self.stats.inc_changed,
                                  incumbent=self.incumbent)

        inc_id = self.stats.inc_changed  # ID of incumbent

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

            self.incumbent, inc_perf = self.inten.intensify(
                challengers=challengers,
                incumbent=self.incumbent,
                run_history=self.runhistory,
                objective=self.objective,
                time_bound=max(0.01, time_spend))

            # TODO: Write run history into database
            if self.scenario.shared_model:
                pSMAC.write(run_history=self.runhistory,
                            output_directory=self.scenario.output_dir,
                            num_run=self.num_run)

            if iteration == max_iters:
                break

            iteration += 1

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.get_remaing_time_budget() < 0 or \
                    self.stats.get_remaining_ta_budget() < 0 or \
                    self.stats.get_remaining_ta_runs() < 0:
                break

            if self.stats.inc_changed > inc_id:
                self.trajLogger.add_entry(train_perf=inc_perf,
                                          incumbent_id=self.stats.inc_changed,
                                          incumbent=self.incumbent)
                inc_id = self.stats.inc_changed

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

            configuration, acq_val = self.local_search.maximize(start_point)

            configuration.origin = 'Local Search'
            configs_acq.append((acq_val[0][0], configuration))

        # shuffle for random tie-break
        random.shuffle(configs_acq, self.rng.rand)

        # sort according to acq value
        # and return n best configurations
        configs_acq.sort(reverse=True, key=lambda x: x[0])

        return configs_acq
