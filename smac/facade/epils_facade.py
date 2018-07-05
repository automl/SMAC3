import logging
import os
import typing

import numpy as np

from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run import StatusType
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.default_configuration_design import \
    DefaultConfiguration
from smac.initial_design.random_configuration_design import RandomConfiguration
from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign
from smac.intensification.intensification import Intensifier
from smac.optimizer.epils import EPILS_Solver
from smac.optimizer.objective import average_cost
from smac.optimizer.acquisition import EI, LogEI, AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import LocalSearch
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.base_epm import AbstractEPM
from smac.utils.util_funcs import get_types
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.constants import MAXINT
from smac.utils.io.output_directory import create_output_directory
from smac.configspace import Configuration

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"


class EPILS(object):
    """
    Facade to use EPILS mode

    Attributes
    ----------
    logger
    stats : Stats
    solver : EPILS_Solver
        Optimizer object, see :class:`~smac.optimizer.epils.EPILS_Solver`

    Parameters
    ----------
    scenario: smac.scenario.scenario.Scenario
        Scenario object
    tae_runner: ExecuteTARun or callable
        Callable or implementation of :class:`ExecuteTaRun`. In case a
        callable is passed it will be wrapped by tae.ExecuteTaFunc().
        If not set, tae_runner will be initialized with
        the tae.ExecuteTARunOld()
    runhistory: RunHistory
        runhistory to store all algorithm runs
    intensifier: Intensifier
        intensification object to issue a racing to decide the current
        incumbent
    acquisition_function : AcquisitionFunction
        Object that implements the AbstractAcquisitionFunction. Will use
        EI if not set.
    model : AbstractEPM
        Model that implements train() and predict(). Will use a
        RandomForest if not set.
    runhistory2epm : RunHistory2EMP
        Object that implements the AbstractRunHistory2EPM. If None,
        will use RunHistory2EPM4Cost if objective is cost or
        RunHistory2EPM4LogCost if objective is runtime.
    initial_design: InitialDesign
        initial sampling design
    initial_configurations: typing.List[Configuration]
        list of initial configurations for initial design --
        cannot be used together with initial_design
    stats: Stats
        optional stats object
    rng: np.random.RandomState
        Random number generator
    run_id: int, (default: 1)
        Run ID will be used as subfolder for output_dir.

    """

    def __init__(self,
                 scenario: Scenario,
                 # TODO: once we drop python3.4 add type hint
                 # typing.Union[ExecuteTARun, callable]
                 tae_runner=None,
                 runhistory: RunHistory = None,
                 intensifier: Intensifier = None,
                 acquisition_function: AbstractAcquisitionFunction = None,
                 model: AbstractEPM = None,
                 runhistory2epm: AbstractRunHistory2EPM = None,
                 initial_design: InitialDesign = None,
                 initial_configurations: typing.List[Configuration] = None,
                 stats: Stats = None,
                 rng: np.random.RandomState = None,
                 run_id: int = 1):
        """Constructor"""
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        aggregate_func = average_cost
        self.runhistory = None
        self.trajectory = None

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        self.output_dir = create_output_directory(scenario, run_id)
        scenario.write()

        # initialize empty runhistory
        if runhistory is None:
            runhistory = RunHistory(aggregate_func=aggregate_func)
        # inject aggr_func if necessary
        if runhistory.aggregate_func is None:
            runhistory.aggregate_func = aggregate_func

        # initial random number generator
        num_run, rng = self._get_rng(rng=rng)

        # reset random number generator in config space to draw different
        # random configurations with each seed given to SMAC
        scenario.cs.seed(rng.randint(MAXINT))

        # initial Trajectory Logger
        traj_logger = TrajLogger(
            output_dir=self.output_dir, stats=self.stats)

        # initial EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        if model is None:
            model = RandomForestWithInstances(
                    types=types, bounds=bounds,
                    instance_features=scenario.feature_array,
                    seed=rng.randint(MAXINT),
                    pca_components=scenario.PCA_DIM,
                    num_trees=scenario.rf_num_trees,
                    do_bootstrapping=scenario.rf_do_bootstrapping,
                    ratio_features=scenario.rf_ratio_features,
                    min_samples_split=scenario.rf_min_samples_split,
                    min_samples_leaf=scenario.rf_min_samples_leaf,
                    max_depth=scenario.rf_max_depth)
        # initial acquisition function
        if acquisition_function is None:
            if scenario.run_obj == "runtime":
                acquisition_function = LogEI(model=model)
            else:
                acquisition_function = EI(model=model)
        # inject model if necessary
        if acquisition_function.model is None:
            acquisition_function.model = model

        # initialize optimizer on acquisition function
        local_search = LocalSearch(acquisition_function,
                                   scenario.cs,
                                   max_steps=scenario.sls_max_steps,
                                   n_steps_plateau_walk=scenario.sls_n_steps_plateau_walk)

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        if tae_runner is None:
            tae_runner = ExecuteTARunOld(ta=scenario.ta,
                                         stats=self.stats,
                                         run_obj=scenario.run_obj,
                                         runhistory=runhistory,
                                         par_factor=scenario.par_factor,
                                         cost_for_crash=scenario.cost_for_crash)
        # Second case, the tae_runner is a function to be optimized
        elif callable(tae_runner):
            tae_runner = ExecuteTAFuncDict(ta=tae_runner,
                                           stats=self.stats,
                                           run_obj=scenario.run_obj,
                                           memory_limit=scenario.memory_limit,
                                           runhistory=runhistory,
                                           par_factor=scenario.par_factor,
                                           cost_for_crash=scenario.cost_for_crash)
        # Third case, if it is an ExecuteTaRun we can simply use the
        # instance. Otherwise, the next check raises an exception
        elif not isinstance(tae_runner, ExecuteTARun):
            raise TypeError("Argument 'tae_runner' is %s, but must be "
                            "either a callable or an instance of "
                            "ExecuteTaRun. Passing 'None' will result in the "
                            "creation of target algorithm runner based on the "
                            "call string in the scenario file."
                            % type(tae_runner))

        # Check that overall objective and tae objective are the same
        if tae_runner.run_obj != scenario.run_obj:
            raise ValueError("Objective for the target algorithm runner and "
                             "the scenario must be the same, but are '%s' and "
                             "'%s'" % (tae_runner.run_obj, scenario.run_obj))

        # inject stats if necessary
        if tae_runner.stats is None:
            tae_runner.stats = self.stats
        # inject runhistory if necessary
        if tae_runner.runhistory is None:
            tae_runner.runhistory = runhistory
        # inject cost_for_crash
        if tae_runner.crash_cost != scenario.cost_for_crash:
            tae_runner.crash_cost = scenario.cost_for_crash

        # initialize intensification
        if intensifier is None:
            intensifier = Intensifier(tae_runner=tae_runner,
                                      stats=self.stats,
                                      traj_logger=traj_logger,
                                      rng=rng,
                                      instances=scenario.train_insts,
                                      cutoff=scenario.cutoff,
                                      deterministic=scenario.deterministic,
                                      run_obj_time=scenario.run_obj == "runtime",
                                      always_race_against=scenario.cs.get_default_configuration()
                                      if scenario.always_race_default else None,
                                      instance_specifics=scenario.instance_specific,
                                      minR=scenario.minR,
                                      maxR=scenario.maxR,
                                      adaptive_capping_slackfactor=scenario.intens_adaptive_capping_slackfactor,
                                      min_chall=scenario.intens_min_chall)
        # inject deps if necessary
        if intensifier.tae_runner is None:
            intensifier.tae_runner = tae_runner
        if intensifier.stats is None:
            intensifier.stats = self.stats
        if intensifier.traj_logger is None:
            intensifier.traj_logger = traj_logger

        # initial design
        if initial_design is not None and initial_configurations is not None:
            raise ValueError(
                "Either use initial_design or initial_configurations; but not both")

        if initial_configurations is not None:
            initial_design = MultiConfigInitialDesign(tae_runner=tae_runner,
                                                      scenario=scenario,
                                                      stats=self.stats,
                                                      traj_logger=traj_logger,
                                                      runhistory=runhistory,
                                                      rng=rng,
                                                      configs=initial_configurations,
                                                      intensifier=intensifier,
                                                      aggregate_func=aggregate_func)
        elif initial_design is None:
            if scenario.initial_incumbent == "DEFAULT":
                initial_design = DefaultConfiguration(tae_runner=tae_runner,
                                                      scenario=scenario,
                                                      stats=self.stats,
                                                      traj_logger=traj_logger,
                                                      rng=rng)
            elif scenario.initial_incumbent == "RANDOM":
                initial_design = RandomConfiguration(tae_runner=tae_runner,
                                                     scenario=scenario,
                                                     stats=self.stats,
                                                     traj_logger=traj_logger,
                                                     rng=rng)
            else:
                raise ValueError("Don't know what kind of initial_incumbent "
                                 "'%s' is" % scenario.initial_incumbent)
        # inject deps if necessary
        if initial_design.tae_runner is None:
            initial_design.tae_runner = tae_runner
        if initial_design.scenario is None:
            initial_design.scenario = scenario
        if initial_design.stats is None:
            initial_design.stats = self.stats
        if initial_design.traj_logger is None:
            initial_design.traj_logger = traj_logger

        # initial conversion of runhistory into EPM data
        if runhistory2epm is None:

            num_params = len(scenario.cs.get_hyperparameters())
            if scenario.run_obj == "runtime":

                # if we log the performance data,
                # the RFRImputator will already get
                # log transform data from the runhistory
                cutoff = np.log(scenario.cutoff)
                threshold = np.log(scenario.cutoff *
                                     scenario.par_factor)

                imputor = RFRImputator(rng=rng,
                                       cutoff=cutoff,
                                       threshold=threshold,
                                       model=model,
                                       change_threshold=0.01,
                                       max_iter=2)

                runhistory2epm = RunHistory2EPM4LogCost(
                    scenario=scenario, num_params=num_params,
                    success_states=[StatusType.SUCCESS, ],
                    impute_censored_data=True,
                    impute_state=[StatusType.CAPPED, ],
                    imputor=imputor)

            elif scenario.run_obj == 'quality':
                runhistory2epm = RunHistory2EPM4Cost(scenario=scenario, num_params=num_params,
                                                     success_states=[
                                                         StatusType.SUCCESS, ],
                                                     impute_censored_data=False, impute_state=None)

            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)

        # inject scenario if necessary:
        if runhistory2epm.scenario is None:
            runhistory2epm.scenario = scenario

        self.solver = EPILS_Solver(scenario=scenario,
                                   stats=self.stats,
                                   initial_design=initial_design,
                                   runhistory=runhistory,
                                   runhistory2epm=runhistory2epm,
                                   intensifier=intensifier,
                                   aggregate_func=aggregate_func,
                                   num_run=num_run,
                                   model=model,
                                   acq_optimizer=local_search,
                                   acquisition_func=acquisition_function,
                                   rng=rng)

    def _get_rng(self, rng):
        """
        Initialize random number generator

        If rng is None, initialize a new generator
        If rng is Int, create RandomState from that
        If rng is RandomState, return it

        Parameters
        ----------
        rng: np.random.RandomState|int|None

        Returns
        -------
        int, np.random.RandomState

        """
        # initialize random number generator
        if rng is None:
            self.logger.debug('no rng given, using default seed of 1')
            num_run = 1
            rng = np.random.RandomState(seed=num_run)
        elif isinstance(rng, int):
            num_run = rng
            rng = np.random.RandomState(seed=rng)
        elif isinstance(rng, np.random.RandomState):
            num_run = rng.randint(MAXINT)
            rng = rng
        else:
            raise TypeError('Unknown type %s for argument rng. Only accepts '
                            'None, int or np.random.RandomState' % str(type(rng)))
        return num_run, rng

    def optimize(self):
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        incumbent

        """
        incumbent = None
        try:
            incumbent = self.solver.run()
        finally:
            self.solver.stats.print_stats()
            self.logger.info("Final Incumbent: %s", self.solver.incumbent)
            self.runhistory = self.solver.runhistory
            self.trajectory = self.solver.intensifier.traj_logger.trajectory

            if self.output_dir is not None:
                self.solver.runhistory.save_json(
                    fn=os.path.join(self.output_dir, "runhistory.json")
                )
        return incumbent

    def get_tae_runner(self):
        """
        Returns target algorithm evaluator (TAE) object which can run the
        target algorithm given a configuration

        Returns
        -------
        smac.tae.execute_ta_run.ExecuteTARun

        """
        return self.solver.intensifier.tae_runner

    def get_runhistory(self):
        """
        Returns the runhistory (i.e., all evaluated configurations and
        the results)

        Returns
        -------
        smac.runhistory.runhistory.RunHistory

        """
        if not hasattr(self, 'runhistory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.runhistory

    def get_trajectory(self):
        """
        Returns the trajectory (i.e., all incumbent configurations over time)

        Returns
        -------
        List of entries with the following fields:
        'train_perf', 'incumbent_id', 'incumbent',
        'ta_runs', 'ta_time_used', 'wallclock_time'

        """
        if not hasattr(self, 'trajectory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.trajectory

    def get_X_y(self):
        """
        Simple interface to obtain all data in runhistory in X, y format
        Uses smac.runhistory.runhistory2epm.AbstractRunHistory2EPM.get_X_y()

        Returns
        -------
        X: numpy.ndarray
            matrix of all configurations (+ instance features)
        y numpy.ndarray
            vector of cost values; can include censored runs
        cen: numpy.ndarray
            vector of bools indicating whether the y-value is censored

        """
        return self.solver.rh2EPM.get_X_y(self.runhistory)
