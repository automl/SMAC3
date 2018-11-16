import logging
import os
import typing

import numpy as np

# tae
from smac.tae.execute_ta_run import ExecuteTARun
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run import StatusType
# stats and options
from smac.stats.stats import Stats
from smac.scenario.scenario import Scenario
# runhistory
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, \
    RunHistory2EPM4LogCost, RunHistory2EPM4Cost, \
    RunHistory2EPM4InvScaledCost, RunHistory2EPM4LogScaledCost
# Initial designs
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.default_configuration_design import \
    DefaultConfiguration
from smac.initial_design.random_configuration_design import RandomConfiguration
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.factorial_design import FactorialInitialDesign
from smac.initial_design.sobol_design import SobolDesign 
from smac.initial_design.multi_config_initial_design import \
    MultiConfigInitialDesign
# intensification
from smac.intensification.intensification import Intensifier
# optimizer
from smac.optimizer.smbo import SMBO
from smac.optimizer.objective import average_cost
from smac.optimizer.acquisition import EI, LogEI, AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    RandomConfigurationChooser, ChooserCosineAnnealing, ChooserProb
# epm
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.base_epm import AbstractEPM
# utils
from smac.utils.util_funcs import get_types
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.constants import MAXINT, N_TREES
from smac.utils.util_funcs import get_rng
from smac.utils.io.output_directory import create_output_directory
from smac.configspace import Configuration


__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC(object):
    """
    Facade to use SMAC default mode

    Attributes
    ----------
    logger
    stats : Stats
    solver : SMBO
    runhistory : RunHistory
        List with information about previous runs
    trajectory : list
        List of all incumbents

    """

    def __init__(self,
                 scenario: Scenario,
                 tae_runner: typing.Optional[typing.Union[ExecuteTARun, typing.Callable]]=None,
                 runhistory: typing.Optional[RunHistory]=None,
                 intensifier: typing.Optional[Intensifier]=None,
                 acquisition_function: typing.Optional[AbstractAcquisitionFunction]=None,
                 acquisition_function_optimizer: typing.Optional[AcquisitionFunctionMaximizer]=None,
                 model: typing.Optional[AbstractEPM]=None,
                 runhistory2epm: typing.Optional[AbstractRunHistory2EPM]=None,
                 initial_design: typing.Optional[InitialDesign]=None,
                 initial_configurations: typing.Optional[typing.List[Configuration]]=None,
                 stats: typing.Optional[Stats]=None,
                 restore_incumbent: typing.Optional[Configuration]=None,
                 rng: typing.Optional[typing.Union[np.random.RandomState, int]]=None,
                 smbo_class: typing.Optional[SMBO]=None,
                 run_id: typing.Optional[int]=None,
                 random_configuration_chooser: typing.Optional[RandomConfigurationChooser]=None):
        """
        Constructor

        Parameters
        ----------
        scenario : ~smac.scenario.scenario.Scenario
            Scenario object
        tae_runner : ~smac.tae.execute_ta_run.ExecuteTARun or callable
            Callable or implementation of
            :class:`~smac.tae.execute_ta_run.ExecuteTARun`. In case a
            callable is passed it will be wrapped by
            :class:`~smac.tae.execute_func.ExecuteTAFuncDict`.
            If not set, it will be initialized with the
            :class:`~smac.tae.execute_ta_run_old.ExecuteTARunOld`.
        runhistory : RunHistory
            runhistory to store all algorithm runs
        intensifier : Intensifier
            intensification object to issue a racing to decide the current
            incumbent
        acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction
            Object that implements the :class:`~smac.optimizer.acquisition.AbstractAcquisitionFunction`.
            Will use :class:`~smac.optimizer.acquisition.EI` if not set.
        acquisition_function_optimizer : ~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
            Object that implements the :class:`~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer`.
            Will use :class:`smac.optimizer.ei_optimization.InterleavedLocalAndRandomSearch` if not set.
        model : AbstractEPM
            Model that implements train() and predict(). Will use a
            :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` if not set.
        runhistory2epm : ~smac.runhistory.runhistory2epm.RunHistory2EMP
            Object that implements the AbstractRunHistory2EPM. If None,
            will use :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4Cost`
            if objective is cost or
            :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4LogCost`
            if objective is runtime.
        initial_design : InitialDesign
            initial sampling design
        initial_configurations : typing.List[Configuration]
            list of initial configurations for initial design --
            cannot be used together with initial_design
        stats : Stats
            optional stats object
        rng : np.random.RandomState
            Random number generator
        restore_incumbent : Configuration
            incumbent used if restoring to previous state
        smbo_class : ~smac.optimizer.smbo.SMBO
            Class implementing the SMBO interface which will be used to
            instantiate the optimizer class.
        run_id : int (optional)
            Run ID will be used as subfolder for output_dir. If no ``run_id`` is given, a random ``run_id`` will be
            chosen.
        random_configuration_chooser : ~smac.optimizer.random_configuration_chooser.RandomConfigurationChooser
            How often to choose a random configuration during the intensification procedure.

        """
        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        aggregate_func = average_cost

        self.scenario = scenario
        self.output_dir = ""
        if not restore_incumbent:
            # restore_incumbent is used by the CLI interface which provides a method for restoring a SMAC run given an
            # output directory. This is the default path.
            # initial random number generator
            run_id, rng = get_rng(rng=rng, run_id=run_id, logger=self.logger)
            self.output_dir = create_output_directory(scenario, run_id)
        elif scenario.output_dir is not None:
            run_id, rng = get_rng(rng=rng, run_id=run_id, logger=self.logger)
            # output-directory is created in CLI when restoring from a
            # folder. calling the function again in the facade results in two
            # folders being created: run_X and run_X.OLD. if we are
            # restoring, the output-folder exists already and we omit creating it,
            # but set the self-output_dir to the dir.
            # necessary because we want to write traj to new output-dir in CLI.
            self.output_dir = scenario.output_dir_for_this_run

        if (
            scenario.deterministic is True
            and getattr(scenario, 'tuner_timeout', None) is None
            and scenario.run_obj == 'quality'
        ):
            self.logger.info('Optimizing a deterministic scenario for '
                             'quality without a tuner timeout - will make '
                             'SMAC deterministic!')
            scenario.intensification_percentage = 1e-10
        scenario.write()

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        if self.scenario.run_obj == "runtime" and not self.scenario.transform_y == "LOG":
            self.logger.warn("Runtime as objective automatically activates log(y) transformation")
            self.scenario.transform_y = "LOG"

        # initialize empty runhistory
        if runhistory is None:
            runhistory = RunHistory(aggregate_func=aggregate_func)
        # inject aggr_func if necessary
        if runhistory.aggregate_func is None:
            runhistory.aggregate_func = aggregate_func

        if not random_configuration_chooser:
            random_configuration_chooser = ChooserProb(prob=scenario.rand_prob,
                                                       rng=rng)

        # reset random number generator in config space to draw different
        # random configurations with each seed given to SMAC
        scenario.cs.seed(rng.randint(MAXINT))

        # initial Trajectory Logger
        traj_logger = TrajLogger(output_dir=self.output_dir, stats=self.stats)

        # initial EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        if model is None:
            model = RandomForestWithInstances(types=types,
                                              bounds=bounds,
                                              instance_features=scenario.feature_array,
                                              seed=rng.randint(MAXINT),
                                              pca_components=scenario.PCA_DIM,
                                              log_y=scenario.transform_y in ["LOG", "LOGS"],
                                              num_trees=scenario.rf_num_trees,
                                              do_bootstrapping=scenario.rf_do_bootstrapping,
                                              ratio_features=scenario.rf_ratio_features,
                                              min_samples_split=scenario.rf_min_samples_split,
                                              min_samples_leaf=scenario.rf_min_samples_leaf,
                                              max_depth=scenario.rf_max_depth)
        # initial acquisition function
        if acquisition_function is None:
            if scenario.transform_y in ["LOG", "LOGS"]:
                acquisition_function = LogEI(model=model)
            else:
                acquisition_function = EI(model=model)

        # inject model if necessary
        if acquisition_function.model is None:
            acquisition_function.model = model

        # initialize optimizer on acquisition function
        if acquisition_function_optimizer is None:
            acquisition_function_optimizer = InterleavedLocalAndRandomSearch(
                acquisition_function=acquisition_function,
                config_space=scenario.cs,
                rng=np.random.RandomState(seed=rng.randint(MAXINT)),
                max_steps=scenario.sls_max_steps,
                n_steps_plateau_walk=scenario.sls_n_steps_plateau_walk
            )
        elif not isinstance(
                acquisition_function_optimizer,
                AcquisitionFunctionMaximizer,
        ):
            raise ValueError(
                "Argument 'acquisition_function_optimizer' must be of type"
                "'AcquisitionFunctionMaximizer', but is '%s'" %
                type(acquisition_function_optimizer)
            )

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        if tae_runner is None:
            tae_runner = ExecuteTARunOld(ta=scenario.ta,
                                         stats=self.stats,
                                         run_obj=scenario.run_obj,
                                         runhistory=runhistory,
                                         par_factor=scenario.par_factor,
                                         cost_for_crash=scenario.cost_for_crash,
                                         abort_on_first_run_crash=scenario.abort_on_first_run_crash)
        # Second case, the tae_runner is a function to be optimized
        elif callable(tae_runner):
            tae_runner = ExecuteTAFuncDict(ta=tae_runner,
                                           stats=self.stats,
                                           run_obj=scenario.run_obj,
                                           memory_limit=scenario.memory_limit,
                                           runhistory=runhistory,
                                           par_factor=scenario.par_factor,
                                           cost_for_crash=scenario.cost_for_crash,
                                           abort_on_first_run_crash=scenario.abort_on_first_run_crash)
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
                                      use_ta_time_bound=scenario.use_ta_time,
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
            elif scenario.initial_incumbent == "LHD":
                initial_design = LHDesign(runhistory=runhistory,
                                            intensifier=intensifier,
                                            aggregate_func=aggregate_func,
                                            tae_runner=tae_runner,
                                            scenario=scenario,
                                            stats=self.stats,
                                            traj_logger=traj_logger,
                                            rng=rng)
            elif scenario.initial_incumbent == "FACTORIAL":
                initial_design = FactorialInitialDesign(runhistory=runhistory,
                                                        intensifier=intensifier,
                                                        aggregate_func=aggregate_func,
                                                        tae_runner=tae_runner,
                                                        scenario=scenario,
                                                        stats=self.stats,
                                                        traj_logger=traj_logger,
                                                        rng=rng)
            elif scenario.initial_incumbent == "SOBOL":
                initial_design = SobolDesign(runhistory=runhistory,
                                            intensifier=intensifier,
                                            aggregate_func=aggregate_func,
                                            tae_runner=tae_runner,
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
            if scenario.run_obj == 'runtime':

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
                if scenario.transform_y == "NONE":
                    runhistory2epm = RunHistory2EPM4Cost(scenario=scenario, num_params=num_params,
                                                         success_states=[
                                                             StatusType.SUCCESS,
                                                             StatusType.CRASHED],
                                                         impute_censored_data=False, impute_state=None)
                elif scenario.transform_y == "LOG":
                    runhistory2epm = RunHistory2EPM4LogCost(scenario=scenario, num_params=num_params,
                                                         success_states=[
                                                             StatusType.SUCCESS,
                                                             StatusType.CRASHED],
                                                         impute_censored_data=False, impute_state=None)
                elif scenario.transform_y == "LOGS":
                    runhistory2epm = RunHistory2EPM4LogScaledCost(scenario=scenario, num_params=num_params,
                                                         success_states=[
                                                             StatusType.SUCCESS,
                                                             StatusType.CRASHED],
                                                         impute_censored_data=False, impute_state=None)
                elif scenario.transform_y == "INVS":
                    runhistory2epm = RunHistory2EPM4InvScaledCost(scenario=scenario, num_params=num_params,
                                                         success_states=[
                                                             StatusType.SUCCESS,
                                                             StatusType.CRASHED],
                                                         impute_censored_data=False, impute_state=None)

            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)

        # inject scenario if necessary:
        if runhistory2epm.scenario is None:
            runhistory2epm.scenario = scenario

        smbo_args = {
            'scenario': scenario,
            'stats': self.stats,
            'initial_design': initial_design,
            'runhistory': runhistory,
            'runhistory2epm': runhistory2epm,
            'intensifier': intensifier,
            'aggregate_func': aggregate_func,
            'num_run': run_id,
            'model': model,
            'acq_optimizer': acquisition_function_optimizer,
            'acquisition_func': acquisition_function,
            'rng': rng,
            'restore_incumbent': restore_incumbent,
            'random_configuration_chooser': random_configuration_chooser
        }

        if smbo_class is None:
            self.solver = SMBO(**smbo_args)
        else:
            self.solver = smbo_class(**smbo_args)

    def optimize(self):
        """
        Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        incumbent : Configuration
            Best found configuration

        """
        incumbent = None
        try:
            incumbent = self.solver.run()
        finally:
            self.solver.stats.save()
            self.solver.stats.print_stats()
            self.logger.info("Final Incumbent: %s", self.solver.incumbent)
            if self.solver.incumbent and self.solver.incumbent in self.solver.runhistory.get_all_configs():
                self.logger.info("Estimated cost of incumbent: %f",
                                 self.solver.runhistory.get_cost(self.solver.incumbent))
            self.runhistory = self.solver.runhistory
            self.trajectory = self.solver.intensifier.traj_logger.trajectory

            if self.output_dir is not None:
                self.solver.runhistory.save_json(
                    fn=os.path.join(self.output_dir, "runhistory.json")
                )
        return incumbent

    def validate(self,
                 config_mode: typing.Union[typing.List[Configuration], np.ndarray, str]='inc',
                 instance_mode: typing.Union[typing.List[str], str]='train+test',
                 repetitions: int=1, use_epm: bool=False, n_jobs: int=-1, backend: str='threading'):
        """
        Create validator-object and run validation, using
        scenario-information, runhistory from smbo and tae_runner from intensify

        Parameters
        ----------
        config_mode: str or list<Configuration>
            string or directly a list of Configuration
            str from [def, inc, def+inc, wallclock_time, cpu_time, all]
            time evaluates at cpu- or wallclock-timesteps of:
            [max_time/2^0, max_time/2^1, max_time/2^3, ..., default]
            with max_time being the highest recorded time
        instance_mode: string
            what instances to use for validation, from [train, test, train+test]
        repetitions: int
            number of repetitions in nondeterministic algorithms (in
            deterministic will be fixed to 1)
        use_epm: bool
            whether to use an EPM instead of evaluating all runs with the TAE
        n_jobs: int
            number of parallel processes used by joblib
        backend: string
            what backend to be used by joblib

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs

        """
        return self.solver.validate(config_mode, instance_mode, repetitions,
                                    use_epm, n_jobs, backend)

    def get_tae_runner(self):
        """
        Returns target algorithm evaluator (TAE) object which can run the
        target algorithm given a configuration

        Returns
        -------
        TAE: smac.tae.execute_ta_run.ExecuteTARun

        """
        return self.solver.intensifier.tae_runner

    def get_runhistory(self):
        """
        Returns the runhistory (i.e., all evaluated configurations and
         the results).

        Returns
        -------
        Runhistory: smac.runhistory.runhistory.RunHistory

        """
        if not hasattr(self, 'runhistory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.runhistory

    def get_trajectory(self):
        """
        Returns the trajectory (i.e., all incumbent configurations over
        time).

        Returns
        -------
        Trajectory : List of :class:`~smac.utils.io.traj_logging.TrajEntry`

        """
        if not hasattr(self, 'trajectory'):
            raise ValueError('SMAC was not fitted yet. Call optimize() prior '
                             'to accessing the runhistory.')
        return self.trajectory

    def get_X_y(self):
        """
        Simple interface to obtain all data in runhistory in ``X, y`` format.

        Uses
        :meth:`smac.runhistory.runhistory2epm.AbstractRunHistory2EPM.get_X_y()`.

        Returns
        -------
        X: numpy.ndarray
            matrix of all configurations (+ instance features)
        y: numpy.ndarray
            vector of cost values; can include censored runs
        cen: numpy.ndarray
            vector of bools indicating whether the y-value is censored

        """
        return self.solver.rh2EPM.get_X_y(self.runhistory)
