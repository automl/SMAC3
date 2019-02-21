import inspect
import logging
import os
from typing import  List, Union, Optional, Type, Callable

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
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.factorial_design import FactorialInitialDesign
from smac.initial_design.sobol_design import SobolDesign

# intensification
from smac.intensification.intensification import Intensifier
# optimizer
from smac.optimizer.smbo import SMBO
from smac.optimizer.objective import average_cost
from smac.optimizer.acquisition import EI, LogEI, AbstractAcquisitionFunction
from smac.optimizer.ei_optimization import InterleavedLocalAndRandomSearch, \
    AcquisitionFunctionMaximizer
from smac.optimizer.random_configuration_chooser import RandomConfigurationChooser, ChooserProb
# epm
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.base_epm import AbstractEPM
# utils
from smac.utils.util_funcs import get_types
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.constants import MAXINT
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
                 tae_runner: Optional[Union[Type[ExecuteTARun], Callable]] = None,
                 tae_runner_kwargs: Optional[dict] = None,
                 runhistory: Optional[Union[Type[RunHistory], RunHistory]] = None,
                 runhistory_kwargs: Optional[dict] = None,
                 intensifier: Optional[Type[Intensifier]] = None,
                 intensifier_kwargs: Optional[dict] = None,
                 acquisition_function: Optional[Type[AbstractAcquisitionFunction]] = None,
                 acquisition_function_kwargs: Optional[dict] = None,
                 acquisition_function_optimizer: Optional[Type[AcquisitionFunctionMaximizer]] = None,
                 acquisition_function_optimizer_kwargs: Optional[dict] = None,
                 model: Optional[Type[AbstractEPM]] = None,
                 model_kwargs: Optional[dict] = None,
                 runhistory2epm: Optional[Type[AbstractRunHistory2EPM]] = None,
                 runhistory2epm_kwargs: Optional[dict] = None,
                 initial_design: Optional[Type[InitialDesign]] = None,
                 initial_design_kwargs: Optional[dict] = None,
                 initial_configurations: Optional[List[Configuration]] = None,
                 stats: Optional[Stats] = None,
                 restore_incumbent: Optional[Configuration] = None,
                 rng: Optional[Union[np.random.RandomState, int]] = None,
                 smbo_class: Optional[SMBO] = None,
                 run_id: Optional[int] = None,
                 random_configuration_chooser: Optional[Type[RandomConfigurationChooser]] = None,
                 random_configuration_chooser_kwargs: Optional[dict] = None
                 ):
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
        tae_runner_kwargs: Optional[dict]
            arguments passed to constructor of '~tae_runner'
        runhistory : RunHistory
            runhistory to store all algorithm runs
        runhistory_kwargs : Optional[dict]
            arguments passed to constructor of runhistory.
            We strongly advise against changing the aggregation function,
            since it will break some code assumptions
        intensifier : Intensifier
            intensification object to issue a racing to decide the current
            incumbent
        intensifier_kwargs: Optional[dict]
            arguments passed to the constructor of '~intensifier'
        acquisition_function : ~smac.optimizer.acquisition.AbstractAcquisitionFunction
            Class or object that implements the :class:`~smac.optimizer.acquisition.AbstractAcquisitionFunction`.
            Will use :class:`~smac.optimizer.acquisition.EI` or :class:`~smac.optimizer.acquisition.LogEI` if not set.
            `~acquisition_function_kwargs` is passed to the class constructor.
        acquisition_function_kwargs : Optional[dict]
            dictionary to pass specific arguments to ~acquisition_function
        acquisition_function_optimizer : ~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
            Object that implements the :class:`~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer`.
            Will use :class:`smac.optimizer.ei_optimization.InterleavedLocalAndRandomSearch` if not set.
        acquisition_function_optimizer_kwargs: Optional[dict]
            Arguments passed to constructor of '~acquisition_function_optimizer'
        model : AbstractEPM
            Model that implements train() and predict(). Will use a
            :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` if not set.
        model_kwargs : Optional[dict]
            Arguments passed to constructor of '~model'
        runhistory2epm : ~smac.runhistory.runhistory2epm.RunHistory2EMP
            Object that implements the AbstractRunHistory2EPM. If None,
            will use :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4Cost`
            if objective is cost or
            :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4LogCost`
            if objective is runtime.
        runhistory2epm_kwargs: Optional[dict]
            Arguments passed to the constructor of '~runhistory2epm'
        initial_design : InitialDesign
            initial sampling design
        initial_design_kwargs: Optional[dict]
            arguments passed to constructor of `~initial_design'
        initial_configurations : List[Configuration]
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
        random_configuration_chooser_kwargs : Optional[dict]
            arguments of constructor for '~random_configuration_chooser'

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
            self.logger.info('Optimizing a deterministic scenario for quality without a tuner timeout - will make '
                             'SMAC deterministic and only evaluate one configuration per iteration!')
            scenario.intensification_percentage = 1e-10
            scenario.min_chall = 1

        scenario.write()

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        if self.scenario.run_obj == "runtime" and not self.scenario.transform_y == "LOG":
            self.logger.warning("Runtime as objective automatically activates log(y) transformation")
            self.scenario.transform_y = "LOG"

        # initialize empty runhistory
        runhistory_def_kwargs = {'aggregate_func': aggregate_func}
        if runhistory_kwargs is not None:
            runhistory_def_kwargs.update(runhistory_kwargs)
        if runhistory is None:
            runhistory = RunHistory(**runhistory_def_kwargs)
        elif inspect.isclass(runhistory):
            runhistory = runhistory(**runhistory_def_kwargs)
        else:
            if runhistory.aggregate_func is None:
                runhistory.aggregate_func = aggregate_func

        rand_conf_chooser_kwargs = {
           'rng': rng
        }
        if random_configuration_chooser_kwargs is not None:
            rand_conf_chooser_kwargs.update(random_configuration_chooser_kwargs)
        if random_configuration_chooser is None:
            if 'prob' not in rand_conf_chooser_kwargs:
                rand_conf_chooser_kwargs['prob'] = scenario.rand_prob
            random_configuration_chooser = ChooserProb(**rand_conf_chooser_kwargs)
        elif inspect.isclass(random_configuration_chooser):
            random_configuration_chooser = random_configuration_chooser(**rand_conf_chooser_kwargs)
        elif not isinstance(random_configuration_chooser, RandomConfigurationChooser):
            raise ValueError("random_configuration_chooser has to be"
                             " a class or object of RandomConfigurationChooser")

        # reset random number generator in config space to draw different
        # random configurations with each seed given to SMAC
        scenario.cs.seed(rng.randint(MAXINT))

        # initial Trajectory Logger
        traj_logger = TrajLogger(output_dir=self.output_dir, stats=self.stats)

        # initial EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)
        model_def_kwargs = {
            'types': types,
            'bounds': bounds,
            'instance_features': scenario.feature_array,
            'seed': rng.randint(MAXINT),
            'pca_components': scenario.PCA_DIM,
        }
        if model_kwargs is not None:
            model_def_kwargs.update(model_kwargs)
        if model is None:
            for key, value in {
                'log_y': scenario.transform_y in ["LOG", "LOGS"],
                'num_trees': scenario.rf_num_trees,
                'do_bootstrapping': scenario.rf_do_bootstrapping,
                'ratio_features': scenario.rf_ratio_features,
                'min_samples_split': scenario.rf_min_samples_split,
                'min_samples_leaf': scenario.rf_min_samples_leaf,
                'max_depth': scenario.rf_max_depth,
            }.items():
                if key not in model_def_kwargs:
                    model_def_kwargs[key] = value
            model = RandomForestWithInstances(**model_def_kwargs)
        elif inspect.isclass(model):
            model = model(**model_def_kwargs)
        else:
            raise TypeError(
                "Model not recognized: %s" %(type(model)))

        # initial acquisition function
        acq_def_kwargs = {'model': model}
        if acquisition_function_kwargs is not None:
            acq_def_kwargs.update(acquisition_function_kwargs)
        if acquisition_function is None:
            if scenario.transform_y in ["LOG", "LOGS"]:
                acquisition_function = LogEI(**acq_def_kwargs)
            else:
                acquisition_function = EI(**acq_def_kwargs)
        elif inspect.isclass(acquisition_function):
            acquisition_function = acquisition_function(**acq_def_kwargs)
        else:
            raise TypeError(
                "Argument acquisition_function must be None or an object implementing the "
                "AbstractAcquisitionFunction, not %s."
                % type(acquisition_function)
            )

        # initialize optimizer on acquisition function
        acq_func_opt_kwargs = {
            'acquisition_function': acquisition_function,
            'config_space': scenario.cs,
            'rng': rng,
            }
        if acquisition_function_optimizer_kwargs is not None:
            acq_func_opt_kwargs.update(acquisition_function_optimizer_kwargs)
        if acquisition_function_optimizer is None:
            for key, value in {
                'max_steps': scenario.sls_max_steps,
                'n_steps_plateau_walk': scenario.sls_n_steps_plateau_walk,
            }.items():
                if key not in acq_func_opt_kwargs:
                    acq_func_opt_kwargs[key] = value
            acquisition_function_optimizer = InterleavedLocalAndRandomSearch(**acq_func_opt_kwargs)
        elif inspect.isclass(acquisition_function_optimizer):
            acquisition_function_optimizer = acquisition_function_optimizer(**acq_func_opt_kwargs)
        else:
            raise TypeError(
                "Argument acquisition_function_optimizer must be None or an object implementing the "
                "AcquisitionFunctionMaximizer, but is '%s'" %
                type(acquisition_function_optimizer)
            )

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        tae_def_kwargs = {
            'stats': self.stats,
            'run_obj': scenario.run_obj,
            'runhistory': runhistory,
            'par_factor': scenario.par_factor,
            'cost_for_crash': scenario.cost_for_crash,
            'abort_on_first_run_crash': scenario.abort_on_first_run_crash
            }
        if tae_runner_kwargs is not None:
            tae_def_kwargs.update(tae_runner_kwargs)
        if 'ta' not in tae_def_kwargs:
            tae_def_kwargs['ta'] = scenario.ta
        if tae_runner is None:
            tae_def_kwargs['ta'] = scenario.ta
            tae_runner = ExecuteTARunOld(**tae_def_kwargs)
        elif inspect.isclass(tae_runner):
            tae_runner = tae_runner(**tae_def_kwargs)
        elif callable(tae_runner):
            tae_def_kwargs['ta'] = tae_runner
            tae_runner = ExecuteTAFuncDict(**tae_def_kwargs)
        else:
            raise TypeError("Argument 'tae_runner' is %s, but must be "
                            "either None, a callable or an object implementing "
                            "ExecuteTaRun. Passing 'None' will result in the "
                            "creation of target algorithm runner based on the "
                            "call string in the scenario file."
                            % type(tae_runner))

        # Check that overall objective and tae objective are the same
        if tae_runner.run_obj != scenario.run_obj:
            raise ValueError("Objective for the target algorithm runner and "
                             "the scenario must be the same, but are '%s' and "
                             "'%s'" % (tae_runner.run_obj, scenario.run_obj))

        # initialize intensification
        intensifier_def_kwargs = {
            'tae_runner': tae_runner,
            'stats': self.stats,
            'traj_logger': traj_logger,
            'rng': rng,
            'instances': scenario.train_insts,
            'cutoff': scenario.cutoff,
            'deterministic': scenario.deterministic,
            'run_obj_time': scenario.run_obj == "runtime",
            'always_race_against': scenario.cs.get_default_configuration()
                                   if scenario.always_race_default else None,
            'use_ta_time_bound': scenario.use_ta_time,
            'instance_specifics': scenario.instance_specific,
            'minR': scenario.minR,
            'maxR': scenario.maxR,
            'adaptive_capping_slackfactor': scenario.intens_adaptive_capping_slackfactor,
            'min_chall': scenario.intens_min_chall
            }
        if intensifier_kwargs is not None:
            intensifier_def_kwargs.update(intensifier_kwargs)
        if intensifier is None:
            intensifier = Intensifier(**intensifier_def_kwargs)
        elif inspect.isclass(intensifier):
            intensifier = intensifier(**intensifier_def_kwargs)
        else:
            raise TypeError(
                "Argument intensifier must be None or an object implementing the Intensifier, but is '%s'" %
                type(intensifier)
            )

        # initial design
        if initial_design is not None and initial_configurations is not None:
            raise ValueError(
                "Either use initial_design or initial_configurations; but not both")

        init_design_def_kwargs = {
            'tae_runner': tae_runner,
            'scenario': scenario,
            'stats': self.stats,
            'traj_logger': traj_logger,
            'runhistory': runhistory,
            'rng': rng,
            'configs': initial_configurations,
            'intensifier': intensifier,
            'aggregate_func': aggregate_func,
            'n_configs_x_params': 0,
            'max_config_fracs': 0.0
            }
        if initial_design_kwargs is not None:
            init_design_def_kwargs.update(initial_design_kwargs)
        if initial_configurations is not None:
            initial_design = InitialDesign(**init_design_def_kwargs)
        elif initial_design is None:
            if scenario.initial_incumbent == "DEFAULT":
                init_design_def_kwargs['max_config_fracs'] = 0.0
                initial_design = DefaultConfiguration(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "RANDOM":
                init_design_def_kwargs['max_config_fracs'] = 0.0
                initial_design = RandomConfigurations(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "LHD":
                initial_design = LHDesign(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "FACTORIAL":
                initial_design = FactorialInitialDesign(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "SOBOL":
                initial_design = SobolDesign(**init_design_def_kwargs)
            else:
                raise ValueError("Don't know what kind of initial_incumbent "
                                 "'%s' is" % scenario.initial_incumbent)
        elif inspect.isclass(initial_design):
            initial_design = initial_design(**init_design_def_kwargs)
        else:
            raise TypeError(
                "Argument initial_design must be None or an object implementing the InitialDesign, but is '%s'" %
                type(initial_design)
            )

        # if we log the performance data,
        # the RFRImputator will already get
        # log transform data from the runhistory
        if scenario.transform_y in ["LOG", "LOGS"]:
            cutoff = np.log(np.nanmin([np.inf, np.float_(scenario.cutoff)]))
            threshold = cutoff + np.log(scenario.par_factor)
        else:
            cutoff = np.nanmin([np.inf, np.float_(scenario.cutoff)])
            threshold = cutoff * scenario.par_factor
        num_params = len(scenario.cs.get_hyperparameters())
        imputor = RFRImputator(rng=rng,
                               cutoff=cutoff,
                               threshold=threshold,
                               model=model,
                               change_threshold=0.01,
                               max_iter=2)

        r2e_def_kwargs = {
            'scenario': scenario,
            'num_params': num_params,
            'success_states': [StatusType.SUCCESS, ],
            'impute_censored_data': True,
            'impute_state': [StatusType.CAPPED, ],
            'imputor': imputor,
            'scale_perc': 5
            }
        if scenario.run_obj == 'quality':
            r2e_def_kwargs.update({
                'success_states': [StatusType.SUCCESS, StatusType.CRASHED],
                'impute_censored_data': False,
                'impute_state': None,
            })
        if runhistory2epm_kwargs is not None:
            r2e_def_kwargs.update(runhistory2epm_kwargs)
        if runhistory2epm is None:
            if scenario.run_obj == 'runtime':
                runhistory2epm = RunHistory2EPM4LogCost(**r2e_def_kwargs)
            elif scenario.run_obj == 'quality':
                if scenario.transform_y == "NONE":
                    runhistory2epm = RunHistory2EPM4Cost(**r2e_def_kwargs)
                elif scenario.transform_y == "LOG":
                    runhistory2epm = RunHistory2EPM4LogCost(**r2e_def_kwargs)
                elif scenario.transform_y == "LOGS":
                    runhistory2epm = RunHistory2EPM4LogScaledCost(**r2e_def_kwargs)
                elif scenario.transform_y == "INVS":
                    runhistory2epm = RunHistory2EPM4InvScaledCost(**r2e_def_kwargs)
            else:
                raise ValueError('Unknown run objective: %s. Should be either '
                                 'quality or runtime.' % self.scenario.run_obj)
        elif inspect.isclass(runhistory2epm):
            runhistory2epm = runhistory2epm(**r2e_def_kwargs)
        else:
            raise TypeError(
                "Argument runhistory2epm must be None or an object implementing the RunHistory2EPM, but is '%s'" %
                type(runhistory2epm)
            )

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
                 config_mode: Union[List[Configuration], np.ndarray, str] = 'inc',
                 instance_mode: Union[List[str], str] = 'train+test',
                 repetitions: int = 1,
                 use_epm: bool = False,
                 n_jobs: int = -1, backend:
                 str = 'threading'):
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
