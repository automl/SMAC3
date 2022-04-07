from typing import Any, Callable, Dict, List, Optional, Type, Union, cast

import inspect
import logging

import dask.distributed  # type: ignore
import joblib  # type: ignore
import numpy as np

from smac.configspace import Configuration
from smac.epm.base_epm import AbstractEPM

# epm
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.rfr_imputator import RFRImputator
from smac.epm.util_funcs import get_rng, get_types
from smac.initial_design.default_configuration_design import DefaultConfiguration
from smac.initial_design.factorial_design import FactorialInitialDesign

# Initial designs
from smac.initial_design.initial_design import InitialDesign
from smac.initial_design.latin_hypercube_design import LHDesign
from smac.initial_design.random_configuration_design import RandomConfigurations
from smac.initial_design.sobol_design import SobolDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.intensification.hyperband import Hyperband

# intensification
from smac.intensification.intensification import Intensifier
from smac.intensification.successive_halving import SuccessiveHalving
from smac.optimizer.acquisition import (
    EI,
    AbstractAcquisitionFunction,
    IntegratedAcquisitionFunction,
    LogEI,
    PriorAcquisitionFunction,
)
from smac.optimizer.ei_optimization import (
    AcquisitionFunctionMaximizer,
    LocalAndSortedPriorRandomSearch,
    LocalAndSortedRandomSearch,
)
from smac.optimizer.multi_objective.abstract_multi_objective_algorithm import (
    AbstractMultiObjectiveAlgorithm,
)
from smac.optimizer.multi_objective.aggregation_strategy import (
    AggregationStrategy,
    MeanAggregationStrategy,
)
from smac.optimizer.random_configuration_chooser import (
    ChooserProb,
    RandomConfigurationChooser,
)

# optimizer
from smac.optimizer.smbo import SMBO

# runhistory
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import (
    AbstractRunHistory2EPM,
    RunHistory2EPM4Cost,
    RunHistory2EPM4InvScaledCost,
    RunHistory2EPM4LogCost,
    RunHistory2EPM4LogScaledCost,
)
from smac.scenario.scenario import Scenario

# stats and options
from smac.stats.stats import Stats
from smac.tae import StatusType

# tae
from smac.tae.base import BaseRunner
from smac.tae.dask_runner import DaskParallelRunner
from smac.tae.execute_func import ExecuteTAFuncDict
from smac.tae.execute_ta_run_old import ExecuteTARunOld
from smac.utils.constants import MAXINT
from smac.utils.io.output_directory import create_output_directory
from smac.utils.io.traj_logging import TrajEntry, TrajLogger

# utils
from smac.utils.logging import format_array

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SMAC4AC(object):
    """Facade to use SMAC default mode for Algorithm configuration.

    Parameters
    ----------
    scenario : ~smac.scenario.scenario.Scenario
        Scenario object
    tae_runner : ~smac.tae.base.BaseRunner or callable
        Callable or implementation of
        :class:`~smac.tae.base.BaseRunner`. In case a
        callable is passed it will be wrapped by
        :class:`~smac.tae.execute_func.ExecuteTAFuncDict`.
        If not set, it will be initialized with the
        :class:`~smac.tae.execute_ta_run_old.ExecuteTARunOld`.
    tae_runner_kwargs: Optional[Dict]
        arguments passed to constructor of '~tae_runner'
    runhistory : RunHistory
        runhistory to store all algorithm runs
    runhistory_kwargs : Optional[Dict]
        arguments passed to constructor of runhistory.
        We strongly advise against changing the aggregation function,
        since it will break some code assumptions
    intensifier : AbstractRacer
        intensification object or class to issue a racing to decide the current
        incumbent. Default: class `Intensifier`
    intensifier_kwargs: Optional[Dict]
        arguments passed to the constructor of '~intensifier'
    acquisition_function : `~smac.optimizer.acquisition.AbstractAcquisitionFunction`
        Class or object that implements the :class:`~smac.optimizer.acquisition.AbstractAcquisitionFunction`.
        Will use :class:`~smac.optimizer.acquisition.EI` or :class:`~smac.optimizer.acquisition.LogEI` if not set.
        `~acquisition_function_kwargs` is passed to the class constructor.
    acquisition_function_kwargs : Optional[Dict]
        dictionary to pass specific arguments to ~acquisition_function
    integrate_acquisition_function : bool, default=False
        Whether to integrate the acquisition function. Works only with models which can sample their
        hyperparameters (i.e. GaussianProcessMCMC).
    user_priors : bool, default=False
        Whether to make use of user priors in the optimization procedure, using PriorAcquisitionFunction.
    user_prior_kwargs : Optional[Dict]
        Dictionary to pass specific arguments to optimization with prior, e.g. prior confidence parameter,
        and the floor value for the prior (lowest possible value the prior can take).
    acquisition_function_optimizer : ~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer
        Object that implements the :class:`~smac.optimizer.ei_optimization.AcquisitionFunctionMaximizer`.
        Will use :class:`smac.optimizer.ei_optimization.InterleavedLocalAndRandomSearch` if not set.
    acquisition_function_optimizer_kwargs: Optional[dict]
        Arguments passed to constructor of `~acquisition_function_optimizer`
    model : AbstractEPM
        Model that implements train() and predict(). Will use a
        :class:`~smac.epm.rf_with_instances.RandomForestWithInstances` if not set.
    model_kwargs : Optional[dict]
        Arguments passed to constructor of `~model`
    runhistory2epm : ~smac.runhistory.runhistory2epm.RunHistory2EMP
        Object that implements the AbstractRunHistory2EPM. If None,
        will use :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4Cost`
        if objective is cost or
        :class:`~smac.runhistory.runhistory2epm.RunHistory2EPM4LogCost`
        if objective is runtime.
    runhistory2epm_kwargs: Optional[dict]
        Arguments passed to the constructor of `~runhistory2epm`
    multi_objective_algorithm: Optional[Type["AbstractMultiObjectiveAlgorithm"]]
        Class that implements multi objective logic. If None, will use:
        smac.optimizer.multi_objective.aggregation_strategy.MeanAggregationStrategy
        Multi objective only becomes active if the objective
        specified in `~scenario.run_obj` is a List[str] with at least two entries.
    multi_objective_kwargs: Optional[Dict]
        Arguments passed to `~multi_objective_algorithm`.
    initial_design : InitialDesign
        initial sampling design
    initial_design_kwargs: Optional[dict]
        arguments passed to constructor of `~initial_design`
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
        arguments of constructor for `~random_configuration_chooser`
    dask_client : dask.distributed.Client
        User-created dask client, can be used to start a dask cluster and then attach SMAC to it.
    n_jobs : int, optional
        Number of jobs. If > 1 or -1, this creates a dask client if ``dask_client`` is ``None``. Will
        be ignored if ``dask_client`` is not ``None``.
        If ``None``, this value will be set to 1, if ``-1``, this will be set to the number of cpu cores.

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

    def __init__(
        self,
        scenario: Scenario,
        tae_runner: Optional[Union[Type[BaseRunner], Callable]] = None,
        tae_runner_kwargs: Optional[Dict] = None,
        runhistory: Optional[Union[Type[RunHistory], RunHistory]] = None,
        runhistory_kwargs: Optional[Dict] = None,
        intensifier: Optional[Type[AbstractRacer]] = None,
        intensifier_kwargs: Optional[Dict] = None,
        acquisition_function: Optional[Type[AbstractAcquisitionFunction]] = None,
        acquisition_function_kwargs: Optional[Dict] = None,
        integrate_acquisition_function: bool = False,
        user_priors: bool = False,
        user_prior_kwargs: Optional[Dict] = None,
        acquisition_function_optimizer: Optional[Type[AcquisitionFunctionMaximizer]] = None,
        acquisition_function_optimizer_kwargs: Optional[Dict] = None,
        model: Optional[Type[AbstractEPM]] = None,
        model_kwargs: Optional[Dict] = None,
        runhistory2epm: Optional[Type[AbstractRunHistory2EPM]] = None,
        runhistory2epm_kwargs: Optional[Dict] = None,
        multi_objective_algorithm: Optional[Type[AbstractMultiObjectiveAlgorithm]] = None,
        multi_objective_kwargs: Optional[Dict] = None,
        initial_design: Optional[Type[InitialDesign]] = None,
        initial_design_kwargs: Optional[Dict] = None,
        initial_configurations: Optional[List[Configuration]] = None,
        stats: Optional[Stats] = None,
        restore_incumbent: Optional[Configuration] = None,
        rng: Optional[Union[np.random.RandomState, int]] = None,
        smbo_class: Optional[Type[SMBO]] = None,
        run_id: Optional[int] = None,
        random_configuration_chooser: Optional[Type[RandomConfigurationChooser]] = None,
        random_configuration_chooser_kwargs: Optional[Dict] = None,
        dask_client: Optional[dask.distributed.Client] = None,
        n_jobs: Optional[int] = 1,
    ):
        self.logger = logging.getLogger(self.__module__ + "." + self.__class__.__name__)

        self.scenario = scenario
        self.output_dir = ""
        if not restore_incumbent:
            # restore_incumbent is used by the CLI interface which provides a method for restoring a SMAC run given an
            # output directory. This is the default path.
            # initial random number generator
            run_id, rng = get_rng(rng=rng, run_id=run_id, logger=self.logger)
            self.output_dir = create_output_directory(scenario, run_id)
        elif scenario.output_dir is not None:  # type: ignore[attr-defined] # noqa F821
            run_id, rng = get_rng(rng=rng, run_id=run_id, logger=self.logger)
            # output-directory is created in CLI when restoring from a
            # folder. calling the function again in the facade results in two
            # folders being created: run_X and run_X.OLD. if we are
            # restoring, the output-folder exists already and we omit creating it,
            # but set the self-output_dir to the dir.
            # necessary because we want to write traj to new output-dir in CLI.
            self.output_dir = cast(str, scenario.output_dir_for_this_run)  # type: ignore[attr-defined] # noqa F821
        rng = cast(np.random.RandomState, rng)

        if (
            scenario.deterministic is True  # type: ignore[attr-defined] # noqa F821
            and getattr(scenario, "tuner_timeout", None) is None
            and scenario.run_obj == "quality"  # type: ignore[attr-defined] # noqa F821
        ):
            self.logger.info(
                "Optimizing a deterministic scenario for quality without a tuner timeout - will make "
                "SMAC deterministic and only evaluate one configuration per iteration!"
            )
            scenario.intensification_percentage = 1e-10  # type: ignore[attr-defined] # noqa F821
            scenario.min_chall = 1  # type: ignore[attr-defined] # noqa F821

        scenario.write()

        # initialize stats object
        if stats:
            self.stats = stats
        else:
            self.stats = Stats(scenario)

        if self.scenario.run_obj == "runtime" and not self.scenario.transform_y == "LOG":  # type: ignore[attr-defined] # noqa F821
            self.logger.warning("Runtime as objective automatically activates log(y) transformation")
            self.scenario.transform_y = "LOG"  # type: ignore[attr-defined] # noqa F821

        # initialize empty runhistory
        num_obj = len(scenario.multi_objectives)  # type: ignore[attr-defined] # noqa F821
        runhistory_def_kwargs = {}
        if runhistory_kwargs is not None:
            runhistory_def_kwargs.update(runhistory_kwargs)
        if runhistory is None:
            runhistory = RunHistory(**runhistory_def_kwargs)
        elif inspect.isclass(runhistory):
            runhistory = runhistory(**runhistory_def_kwargs)  # type: ignore[operator] # noqa F821
        elif isinstance(runhistory, RunHistory):
            pass
        else:
            raise ValueError("runhistory has to be a class or an object of RunHistory")

        rand_conf_chooser_kwargs = {"rng": rng}
        if random_configuration_chooser_kwargs is not None:
            rand_conf_chooser_kwargs.update(random_configuration_chooser_kwargs)
        if random_configuration_chooser is None:
            if "prob" not in rand_conf_chooser_kwargs:
                rand_conf_chooser_kwargs["prob"] = scenario.rand_prob  # type: ignore[attr-defined] # noqa F821
            random_configuration_chooser_instance = ChooserProb(
                **rand_conf_chooser_kwargs  # type: ignore[arg-type] # noqa F821  # type: RandomConfigurationChooser
            )
        elif inspect.isclass(random_configuration_chooser):
            random_configuration_chooser_instance = random_configuration_chooser(  # type: ignore # noqa F821
                **rand_conf_chooser_kwargs  # type: ignore[arg-type] # noqa F821
            )
        elif not isinstance(random_configuration_chooser, RandomConfigurationChooser):
            raise ValueError(
                "random_configuration_chooser has to be" " a class or object of RandomConfigurationChooser"
            )

        # reset random number generator in config space to draw different
        # random configurations with each seed given to SMAC
        scenario.cs.seed(rng.randint(MAXINT))  # type: ignore[attr-defined] # noqa F821

        # initial Trajectory Logger
        traj_logger = TrajLogger(output_dir=self.output_dir, stats=self.stats)

        # initial EPM
        types, bounds = get_types(scenario.cs, scenario.feature_array)  # type: ignore[attr-defined] # noqa F821
        model_def_kwargs = {
            "types": types,
            "bounds": bounds,
            "instance_features": scenario.feature_array,
            "seed": rng.randint(MAXINT),
            "pca_components": scenario.PCA_DIM,
        }
        if model_kwargs is not None:
            model_def_kwargs.update(model_kwargs)
        if model is None:
            for key, value in {
                "log_y": scenario.transform_y in ["LOG", "LOGS"],  # type: ignore[attr-defined] # noqa F821
                "num_trees": scenario.rf_num_trees,  # type: ignore[attr-defined] # noqa F821
                "do_bootstrapping": scenario.rf_do_bootstrapping,  # type: ignore[attr-defined] # noqa F821
                "ratio_features": scenario.rf_ratio_features,  # type: ignore[attr-defined] # noqa F821
                "min_samples_split": scenario.rf_min_samples_split,  # type: ignore[attr-defined] # noqa F821
                "min_samples_leaf": scenario.rf_min_samples_leaf,  # type: ignore[attr-defined] # noqa F821
                "max_depth": scenario.rf_max_depth,  # type: ignore[attr-defined] # noqa F821
            }.items():
                if key not in model_def_kwargs:
                    model_def_kwargs[key] = value
            model_def_kwargs["configspace"] = self.scenario.cs  # type: ignore[attr-defined] # noqa F821
            model_instance = RandomForestWithInstances(
                **model_def_kwargs  # type: ignore[arg-type] # noqa F821  # type: AbstractEPM
            )
        elif inspect.isclass(model):
            model_def_kwargs["configspace"] = self.scenario.cs  # type: ignore[attr-defined] # noqa F821
            model_instance = model(**model_def_kwargs)  # type: ignore # noqa F821
        else:
            raise TypeError("Model not recognized: %s" % (type(model)))

        # initial acquisition function
        acq_def_kwargs = {"model": model_instance}
        if acquisition_function_kwargs is not None:
            acq_def_kwargs.update(acquisition_function_kwargs)

        acquisition_function_instance = None  # type: Optional[AbstractAcquisitionFunction]
        if acquisition_function is None:
            if scenario.transform_y in ["LOG", "LOGS"]:  # type: ignore[attr-defined] # noqa F821
                acquisition_function_instance = LogEI(**acq_def_kwargs)  # type: ignore[arg-type] # noqa F821
            else:
                acquisition_function_instance = EI(**acq_def_kwargs)  # type: ignore[arg-type] # noqa F821
        elif inspect.isclass(acquisition_function):
            acquisition_function_instance = acquisition_function(**acq_def_kwargs)
        else:
            raise TypeError(
                "Argument acquisition_function must be None or an object implementing the "
                "AbstractAcquisitionFunction, not %s." % type(acquisition_function)
            )
        if integrate_acquisition_function:
            acquisition_function_instance = IntegratedAcquisitionFunction(
                acquisition_function=acquisition_function_instance,  # type: ignore
                **acq_def_kwargs,
            )

        if user_priors:
            if user_prior_kwargs is None:
                user_prior_kwargs = {}

            # a solid default value for decay_beta - empirically founded
            default_beta = scenario.ta_run_limit / 10  # type: ignore
            discretize = isinstance(model_instance, (RandomForestWithInstances, RFRImputator))
            user_prior_kwargs["decay_beta"] = user_prior_kwargs.get("decay_beta", default_beta)
            user_prior_kwargs["discretize"] = discretize

            acquisition_function_instance = PriorAcquisitionFunction(
                acquisition_function=acquisition_function_instance,  # type: ignore
                **user_prior_kwargs,
                **acq_def_kwargs,  # type: ignore
            )
            acquisition_function_optimizer = LocalAndSortedPriorRandomSearch

        # initialize optimizer on acquisition function
        acq_func_opt_kwargs = {
            "acquisition_function": acquisition_function_instance,
            "config_space": scenario.cs,  # type: ignore[attr-defined] # noqa F821
            "rng": rng,
        }
        if user_priors:
            acq_func_opt_kwargs["uniform_config_space"] = scenario.cs.remove_hyperparameter_priors()  # type: ignore

        if acquisition_function_optimizer_kwargs is not None:
            acq_func_opt_kwargs.update(acquisition_function_optimizer_kwargs)
        if acquisition_function_optimizer is None:
            for key, value in {
                "max_steps": scenario.sls_max_steps,  # type: ignore[attr-defined] # noqa F821
                "n_steps_plateau_walk": scenario.sls_n_steps_plateau_walk,  # type: ignore[attr-defined] # noqa F821
            }.items():
                if key not in acq_func_opt_kwargs:
                    acq_func_opt_kwargs[key] = value
            acquisition_function_optimizer_instance = LocalAndSortedRandomSearch(**acq_func_opt_kwargs)  # type: ignore
        elif inspect.isclass(acquisition_function_optimizer):
            acquisition_function_optimizer_instance = acquisition_function_optimizer(  # type: ignore # noqa F821
                **acq_func_opt_kwargs
            )  # type: ignore # noqa F821
        else:
            raise TypeError(
                "Argument acquisition_function_optimizer must be None or an object implementing the "
                "AcquisitionFunctionMaximizer, but is '%s'" % type(acquisition_function_optimizer)
            )

        # initialize tae_runner
        # First case, if tae_runner is None, the target algorithm is a call
        # string in the scenario file
        tae_def_kwargs = {
            "stats": self.stats,
            "run_obj": scenario.run_obj,
            "par_factor": scenario.par_factor,  # type: ignore[attr-defined] # noqa F821
            "cost_for_crash": scenario.cost_for_crash,  # type: ignore[attr-defined] # noqa F821
            "abort_on_first_run_crash": scenario.abort_on_first_run_crash,  # type: ignore[attr-defined] # noqa F821
            "multi_objectives": scenario.multi_objectives,  # type: ignore[attr-defined] # noqa F821
        }
        if tae_runner_kwargs is not None:
            tae_def_kwargs.update(tae_runner_kwargs)

        if "ta" not in tae_def_kwargs:
            tae_def_kwargs["ta"] = scenario.ta  # type: ignore[attr-defined] # noqa F821
        if tae_runner is None:
            tae_def_kwargs["ta"] = scenario.ta  # type: ignore[attr-defined] # noqa F821
            tae_runner_instance = ExecuteTARunOld(
                **tae_def_kwargs
            )  # type: ignore[arg-type] # noqa F821  # type: BaseRunner
        elif inspect.isclass(tae_runner):
            tae_runner_instance = cast(BaseRunner, tae_runner(**tae_def_kwargs))  # type: ignore
        elif callable(tae_runner):
            tae_def_kwargs["ta"] = tae_runner
            tae_def_kwargs["use_pynisher"] = scenario.limit_resources  # type: ignore[attr-defined] # noqa F821
            tae_def_kwargs["memory_limit"] = scenario.memory_limit  # type: ignore[attr-defined] # noqa F821
            tae_runner_instance = ExecuteTAFuncDict(**tae_def_kwargs)  # type: ignore
        else:
            raise TypeError(
                "Argument 'tae_runner' is %s, but must be "
                "either None, a callable or an object implementing "
                "BaseRunner. Passing 'None' will result in the "
                "creation of target algorithm runner based on the "
                "call string in the scenario file." % type(tae_runner)
            )

        # In case of a parallel run, wrap the single worker in a parallel
        # runner
        if n_jobs is None or n_jobs == 1:
            _n_jobs = 1
        elif n_jobs == -1:
            _n_jobs = joblib.cpu_count()
        elif n_jobs > 0:
            _n_jobs = n_jobs
        else:
            raise ValueError("Number of tasks must be positive, None or -1, but is %s" % str(n_jobs))
        if _n_jobs > 1 or dask_client is not None:
            tae_runner_instance = DaskParallelRunner(  # type: ignore
                tae_runner_instance,
                n_workers=_n_jobs,
                output_directory=self.output_dir,
                dask_client=dask_client,
            )

        # Check that overall objective and tae objective are the same
        # TODO: remove these two ignores once the scenario object knows all its attributes!
        if tae_runner_instance.run_obj != scenario.run_obj:  # type: ignore[union-attr] # noqa F821
            raise ValueError(
                "Objective for the target algorithm runner and "
                "the scenario must be the same, but are '%s' and "
                "'%s'" % (tae_runner_instance.run_obj, scenario.run_obj)
            )  # type: ignore[union-attr] # noqa F821

        if intensifier is None:
            intensifier = Intensifier

        if isinstance(intensifier, AbstractRacer):
            intensifier_instance = intensifier
        elif inspect.isclass(intensifier):
            # initialize intensification
            intensifier_def_kwargs = {
                "stats": self.stats,
                "traj_logger": traj_logger,
                "rng": rng,
                "instances": scenario.train_insts,  # type: ignore[attr-defined] # noqa F821
                "cutoff": scenario.cutoff,  # type: ignore[attr-defined] # noqa F821
                "deterministic": scenario.deterministic,  # type: ignore[attr-defined] # noqa F821
                "run_obj_time": scenario.run_obj == "runtime",  # type: ignore[attr-defined] # noqa F821
                "instance_specifics": scenario.instance_specific,  # type: ignore[attr-defined] # noqa F821
                "adaptive_capping_slackfactor": scenario.intens_adaptive_capping_slackfactor,  # type: ignore[attr-defined] # noqa F821
                "min_chall": scenario.intens_min_chall,  # type: ignore[attr-defined] # noqa F821
            }

            if issubclass(intensifier, Intensifier):
                intensifier_def_kwargs["always_race_against"] = scenario.cs.get_default_configuration()  # type: ignore[attr-defined] # noqa F821
                intensifier_def_kwargs["use_ta_time_bound"] = scenario.use_ta_time  # type: ignore[attr-defined] # noqa F821
                intensifier_def_kwargs["minR"] = scenario.minR  # type: ignore[attr-defined] # noqa F821
                intensifier_def_kwargs["maxR"] = scenario.maxR  # type: ignore[attr-defined] # noqa F821

            if intensifier_kwargs is not None:
                intensifier_def_kwargs.update(intensifier_kwargs)

            intensifier_instance = intensifier(**intensifier_def_kwargs)  # type: ignore[arg-type] # noqa F821
        else:
            raise TypeError(
                "Argument intensifier must be None or an object implementing the AbstractRacer, but is '%s'"
                % type(intensifier)
            )

        # initialize multi objective
        # the multi_objective_algorithm_instance will be passed to the runhistory2epm object
        multi_objective_algorithm_instance = None  # type: Optional[AbstractMultiObjectiveAlgorithm]

        if scenario.multi_objectives is not None and num_obj > 1:  # type: ignore[attr-defined] # noqa F821
            # define any defaults here
            _multi_objective_kwargs = {"rng": rng, "num_obj": num_obj}

            if multi_objective_kwargs is not None:
                _multi_objective_kwargs.update(multi_objective_kwargs)

            if multi_objective_algorithm is None:
                multi_objective_algorithm_instance = MeanAggregationStrategy(
                    **_multi_objective_kwargs  # type: ignore[arg-type] # noqa F821
                )
            elif inspect.isclass(multi_objective_algorithm):
                multi_objective_algorithm_instance = multi_objective_algorithm(**_multi_objective_kwargs)
            else:
                raise TypeError("Multi-objective algorithm not recognized: %s" % (type(multi_objective_algorithm)))

        # initial design
        if initial_design is not None and initial_configurations is not None:
            raise ValueError("Either use initial_design or initial_configurations; but not both")

        init_design_def_kwargs = {
            "cs": scenario.cs,  # type: ignore[attr-defined] # noqa F821
            "traj_logger": traj_logger,
            "rng": rng,
            "ta_run_limit": scenario.ta_run_limit,  # type: ignore[attr-defined] # noqa F821
            "configs": initial_configurations,
            "n_configs_x_params": 0,
            "max_config_fracs": 0.0,
        }

        if initial_design_kwargs is not None:
            init_design_def_kwargs.update(initial_design_kwargs)
        if initial_configurations is not None:
            initial_design_instance = InitialDesign(**init_design_def_kwargs)
        elif initial_design is None:
            if scenario.initial_incumbent == "DEFAULT":  # type: ignore[attr-defined] # noqa F821
                init_design_def_kwargs["max_config_fracs"] = 0.0
                initial_design_instance = DefaultConfiguration(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "RANDOM":  # type: ignore[attr-defined] # noqa F821
                init_design_def_kwargs["max_config_fracs"] = 0.0
                initial_design_instance = RandomConfigurations(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "LHD":  # type: ignore[attr-defined] # noqa F821
                initial_design_instance = LHDesign(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "FACTORIAL":  # type: ignore[attr-defined] # noqa F821
                initial_design_instance = FactorialInitialDesign(**init_design_def_kwargs)
            elif scenario.initial_incumbent == "SOBOL":  # type: ignore[attr-defined] # noqa F821
                initial_design_instance = SobolDesign(**init_design_def_kwargs)
            else:
                raise ValueError(
                    "Don't know what kind of initial_incumbent " "'%s' is" % scenario.initial_incumbent  # type: ignore
                )  # type: ignore[attr-defined] # noqa F821
        elif inspect.isclass(initial_design):
            initial_design_instance = initial_design(**init_design_def_kwargs)
        else:
            raise TypeError(
                "Argument initial_design must be None or an object implementing the InitialDesign, but is '%s'"
                % type(initial_design)
            )

        # if we log the performance data,
        # the RFRImputator will already get
        # log transform data from the runhistory
        if scenario.transform_y in ["LOG", "LOGS"]:  # type: ignore[attr-defined] # noqa F821
            cutoff = np.log(np.nanmin([np.inf, np.float_(scenario.cutoff)]))  # type: ignore[arg-type, attr-defined] # noqa F821
            threshold = cutoff + np.log(scenario.par_factor)  # type: ignore[attr-defined] # noqa F821
        else:
            cutoff = np.nanmin([np.inf, np.float_(scenario.cutoff)])  # type: ignore[arg-type, attr-defined] # noqa F821
            threshold = cutoff * scenario.par_factor  # type: ignore[attr-defined] # noqa F821

        num_params = len(scenario.cs.get_hyperparameters())  # type: ignore[attr-defined] # noqa F821
        imputor = RFRImputator(
            rng=rng,
            cutoff=cutoff,
            threshold=threshold,
            model=model_instance,
            change_threshold=0.01,
            max_iter=2,
        )

        r2e_def_kwargs = {
            "scenario": scenario,
            "num_params": num_params,
            "success_states": [
                StatusType.SUCCESS,
            ],
            "impute_censored_data": True,
            "impute_state": [
                StatusType.CAPPED,
            ],
            "imputor": imputor,
            "scale_perc": 5,
        }

        # TODO: consider other sorts of multi-objective algorithms
        if isinstance(multi_objective_algorithm_instance, AggregationStrategy):
            r2e_def_kwargs.update({"multi_objective_algorithm": multi_objective_algorithm_instance})

        if scenario.run_obj == "quality":
            r2e_def_kwargs.update(
                {
                    "success_states": [
                        StatusType.SUCCESS,
                        StatusType.CRASHED,
                        StatusType.MEMOUT,
                    ],
                    "impute_censored_data": False,
                    "impute_state": None,
                }
            )

        if isinstance(intensifier_instance, (SuccessiveHalving, Hyperband)) and scenario.run_obj == "quality":
            r2e_def_kwargs.update(
                {
                    "success_states": [
                        StatusType.SUCCESS,
                        StatusType.CRASHED,
                        StatusType.MEMOUT,
                        StatusType.DONOTADVANCE,
                    ],
                    "consider_for_higher_budgets_state": [
                        StatusType.DONOTADVANCE,
                        StatusType.TIMEOUT,
                        StatusType.CRASHED,
                        StatusType.MEMOUT,
                    ],
                }
            )

        if runhistory2epm_kwargs is not None:
            r2e_def_kwargs.update(runhistory2epm_kwargs)
        if runhistory2epm is None:
            if scenario.run_obj == "runtime":
                rh2epm = RunHistory2EPM4LogCost(
                    **r2e_def_kwargs  # type: ignore
                )  # type: ignore[arg-type] # noqa F821  # type: AbstractRunHistory2EPM
            elif scenario.run_obj == "quality":
                if scenario.transform_y == "NONE":  # type: ignore[attr-defined] # noqa F821
                    rh2epm = RunHistory2EPM4Cost(**r2e_def_kwargs)  # type: ignore # noqa F821
                elif scenario.transform_y == "LOG":  # type: ignore[attr-defined] # noqa F821
                    rh2epm = RunHistory2EPM4LogCost(**r2e_def_kwargs)  # type: ignore # noqa F821
                elif scenario.transform_y == "LOGS":  # type: ignore[attr-defined] # noqa F821
                    rh2epm = RunHistory2EPM4LogScaledCost(**r2e_def_kwargs)  # type: ignore # noqa F821
                elif scenario.transform_y == "INVS":  # type: ignore[attr-defined] # noqa F821
                    rh2epm = RunHistory2EPM4InvScaledCost(**r2e_def_kwargs)  # type: ignore # noqa F821
            else:
                raise ValueError(
                    "Unknown run objective: %s. Should be either "
                    "quality or runtime." % self.scenario.run_obj  # type: ignore # noqa F821
                )
        elif inspect.isclass(runhistory2epm):
            rh2epm = runhistory2epm(**r2e_def_kwargs)  # type: ignore # noqa F821
        else:
            raise TypeError(
                "Argument runhistory2epm must be None or an object implementing the RunHistory2EPM, but is '%s'"
                % type(runhistory2epm)
            )

        smbo_args = {
            "scenario": scenario,
            "stats": self.stats,
            "initial_design": initial_design_instance,
            "runhistory": runhistory,
            "runhistory2epm": rh2epm,
            "intensifier": intensifier_instance,
            "num_run": run_id,
            "model": model_instance,
            "acq_optimizer": acquisition_function_optimizer_instance,
            "acquisition_func": acquisition_function_instance,
            "rng": rng,
            "restore_incumbent": restore_incumbent,
            "random_configuration_chooser": random_configuration_chooser_instance,
            "tae_runner": tae_runner_instance,
        }  # type: Dict[str, Any]

        if smbo_class is None:
            self.solver = SMBO(**smbo_args)  # type: ignore[arg-type] # noqa F821
        else:
            self.solver = smbo_class(**smbo_args)  # type: ignore[arg-type] # noqa F821

    def optimize(self) -> Configuration:
        """Optimizes the algorithm provided in scenario (given in constructor)

        Returns
        -------
        incumbent : Configuration
            Best found configuration
        """
        incumbent = None
        try:
            incumbent = self.solver.run()
        finally:
            self.solver.save()

            self.solver.stats.print_stats()
            self.logger.info("Final Incumbent: %s", self.solver.incumbent)
            if self.solver.incumbent and self.solver.incumbent in self.solver.runhistory.get_all_configs():
                self.logger.info(
                    f"Estimated cost of incumbent: "
                    f"{format_array(self.solver.runhistory.get_cost(self.solver.incumbent))}"
                )
            self.runhistory = self.solver.runhistory
            self.trajectory = self.solver.intensifier.traj_logger.trajectory

        return incumbent

    def validate(
        self,
        config_mode: Union[List[Configuration], str] = "inc",
        instance_mode: Union[List[str], str] = "train+test",
        repetitions: int = 1,
        use_epm: bool = False,
        n_jobs: int = -1,
        backend: str = "threading",
    ) -> RunHistory:
        """Create validator-object and run validation, using scenario- information, runhistory from
        smbo and tae_runner from intensify.

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
        return self.solver.validate(config_mode, instance_mode, repetitions, use_epm, n_jobs, backend)

    def get_tae_runner(self) -> BaseRunner:
        """Returns target algorithm evaluator (TAE) object which can run the target algorithm given
        a configuration.

        Returns
        -------
        TAE: smac.tae.base.BaseRunner
        """
        return self.solver.tae_runner

    def get_runhistory(self) -> RunHistory:
        """Returns the runhistory (i.e., all evaluated configurations and the results).

        Returns
        -------
        Runhistory: smac.runhistory.runhistory.RunHistory
        """
        if not hasattr(self, "runhistory"):
            raise ValueError("SMAC was not fitted yet. Call optimize() prior " "to accessing the runhistory.")
        return self.runhistory

    def get_trajectory(self) -> List[TrajEntry]:
        """Returns the trajectory (i.e., all incumbent configurations over time).

        Returns
        -------
        Trajectory : List of :class:`~smac.utils.io.traj_logging.TrajEntry`
        """
        if not hasattr(self, "trajectory"):
            raise ValueError("SMAC was not fitted yet. Call optimize() prior " "to accessing the runhistory.")
        return self.trajectory

    def register_callback(self, callback: Callable) -> None:
        """Register a callback function.

        Callbacks must implement a class in ``smac.callbacks`` and be instantiated objects.
        They will automatically be registered within SMAC based on which callback class from
        ``smac.callbacks`` they implement.

        Parameters
        ----------
        callback - Callable

        Returns
        -------
        None
        """
        types_to_check = callback.__class__.__mro__
        key = None
        for type_to_check in types_to_check:
            key = self.solver._callback_to_key.get(type_to_check)
            if key is not None:
                break
        if key is None:
            raise ValueError("Cannot register callback of type %s" % type(callback))
        self.solver._callbacks[key].append(callback)
