import logging
import numpy as np
import random
import sys

from ConfigSpace.io import pcs
from ConfigSpace.hyperparameters import CategoricalHyperparameter, \
    UniformFloatHyperparameter, UniformIntegerHyperparameter, Constant

from smac.smbo.acquisition import EI
from smac.smbo.base_solver import BaseSolver
from smac.smbo.rf_with_instances import RandomForestWithInstances
from smac.smbo.local_search import LocalSearch
from smac.smbo.intensification import Intensifier
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import RunHistory2EPM
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats

MAXINT = 2**31 - 1

__author__ = "Aaron Klein, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
#__maintainer__ = "???"
#__email__ = "???"
__version__ = "0.0.1"


class SMBO(BaseSolver):

    def __init__(self, scenario, rng=None):
        '''
        Interface that contains the main Bayesian optimization loop

        Parameters
        ----------
        scenario: smac.scenario.scenario.Scenario
            Scenario object
        rng: numpy.random.RandomState
            Random number generator
        '''
        self.logger = logging.getLogger("smbo")

        if rng is None:
            self.rng = np.random.RandomState(seed=np.random.randint(10000))
        else:
            self.rng = rng

        self.scenario = scenario
        self.config_space = scenario.cs

        # Extract types vector for rf from config space
        self.types = np.zeros(len(self.config_space.get_hyperparameters()),
                              dtype=np.uint)

        # Extract bounds of the input space
        X_lower = np.zeros([self.types.shape[0]])
        X_upper = np.zeros([self.types.shape[0]])

        for i, param in enumerate(self.config_space.get_hyperparameters()):
            if isinstance(param, (UniformFloatHyperparameter,
                                  UniformIntegerHyperparameter)):
                X_lower[i] = 0
                X_upper[i] = 1
            elif isinstance(param, (CategoricalHyperparameter)):
                n_cats = len(param.choices)
                self.types[i] = n_cats
                X_lower[i] = 0
                X_upper[i] = n_cats
            elif isinstance(param, Constant):
                # for constants we simply set types to 0
                # which makes it a numerical parameter
                self.types[i] = 0
                # and we leave the bounds to be 0 for now
            else:
                raise TypeError("Unknown hyperparameter type %s" % type(param))
        self.model = RandomForestWithInstances(self.types,
                                               scenario.feature_array)

        self.acquisition_func = EI(self.model,
                                   X_lower,
                                   X_upper)

        self.local_search = LocalSearch(self.acquisition_func,
                                        self.config_space)
        self.incumbent = None
        self.executor = None

    def run_initial_design(self):
        '''
            runs algorithm runs for a initial design;
            default implementation: running the default configuration on
                                    a random instance-seed pair
            Side effect: adds runs to self.runhistory
        '''

        default_conf = self.config_space.get_default_configuration()
        self.incumbent = default_conf
        rand_inst_id = self.rng.randint(0, len(self.scenario.train_insts))
        # ignore instance specific values
        rand_inst = self.scenario.train_insts[rand_inst_id][0]
        
        if self.scenario.deterministic:
            initial_seed = 0
        else:
            initial_seed = random.randint(0, MAXINT)

        status, cost, runtime, additional_info = self.executor.run(
            default_conf, instance=rand_inst, cutoff=self.scenario.cutoff,
            seed=initial_seed)

        if status in [StatusType.CRASHED or StatusType.ABORT]:
            self.logger.info("First run crashed -- Abort")
            sys.exit(42)

        self.runhistory.add(config=default_conf, cost=cost, time=runtime,
                            status=status,
                            instance_id=rand_inst,
                            seed=initial_seed,
                            additional_info=additional_info)

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
        Stats.start_timing()

        num_params = len(self.config_space.get_hyperparameters())
        self.runhistory = RunHistory()

        # TODO set arguments properly
        rh2EPM = RunHistory2EPM(num_params=num_params,
                                cutoff_time=self.scenario.cutoff,
                                success_states=None,
                                impute_censored_data=False,
                                impute_state=None)

        self.executor = self.scenario.tae_runner

        self.run_initial_design()

        # Main BO loop
        iteration = 1
        while True:

            # TODO: Transform lambda to X

            X, Y = rh2EPM.transform(self.runhistory)

            # TODO: Estimate new configuration
            self.logger.debug("Search for next configuration")
            next_config = self.choose_next(X, Y)

            self.logger.debug("Intensify")
            # TODO: fix timebound of intensifier
            # TODO: add more than one challenger
            inten = Intensifier(executor=self.executor,
                                challengers=[next_config,
                                             self.config_space.sample_configuration()],
                                incumbent=self.incumbent,
                                run_history=self.runhistory,
                                instances=[inst[0]
                                           for inst in self.scenario.train_insts],
                                cutoff=self.scenario.cutoff,
                                deterministic = self.scenario.deterministic,
                                run_obj_time = self.scenario.run_obj == "runtime")

            self.incumbent = inten.intensify()

            # TODO: Write run history into database

            if iteration == max_iters:
                break

            iteration += 1

            if Stats.get_remaing_time_budget() < 0 or Stats.get_remaining_ta_runs() < 0:
                break

        return self.incumbent

    def choose_next(self, X=None, Y=None, n_iters=10):
        """
        Performs one single iteration of Bayesian optimization and estimated
        the next point to evaluate.

        Parameters
        ----------
        X : (N, D) numpy array, optional
            Each column contains a configuration and one set of
            instance features.
        Y : (N, 1) numpy array, optional
            The function values for each configuration instance pair.

        Returns
        -------
        x : (1, H) Configuration Object
            The suggested configuration to evaluate.
        """

        if X is None or Y is None:
            return self.config_space.sample_configuration()

        self.model.train(X, Y)
        self.acquisition_func.update(self.model)

        found_configs = []
        acq_vals = np.zeros([n_iters])

        # Start N local search from different random start points
        for i in range(n_iters):
            if i == 0 and self.incumbent is not None:
                start_point = self.incumbent
            else:
                start_point = self.config_space.sample_configuration()

            configuration, acq_val = self.local_search.maximize(start_point)

            found_configs.append(configuration)
            acq_vals[i] = acq_val[0][0]

        # Return configuration with highest acquisition value
        # TODO JTS: this argmax will always return the first value
        #           if there are multiple results with the same acquisition function
        #           maybe we should randomly tie-break here!
        best = np.argmax(acq_vals)
        # TODO: We could also return a configuration object here, but then also
        # the unit test has to be adapted
        # ML: We have to return the configuration object here or else it is a
        # mess since we cannot convert it back

        return found_configs[best]
