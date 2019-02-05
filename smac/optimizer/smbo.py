import os
import logging
import numpy as np
import time
import typing

import george

from smac.configspace import ConfigurationSpace, Configuration, Constant,\
     CategoricalHyperparameter, UniformFloatHyperparameter, \
     UniformIntegerHyperparameter, InCondition
from smac.configspace.util import convert_configurations_to_array
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gp_default_priors import DefaultPrior
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.intensification import Intensifier
from smac.optimizer import pSMAC
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI, LogEI,\
    LCB, PI
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer, \
    RandomSearch
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM, RunHistory2EPM4LogCost
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.tae.execute_ta_run import FirstRunCrashedException
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator
from smac.utils.util_funcs import get_types
from smac.utils.constants import MAXINT


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
    random_configuration_chooser
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
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration=None,
                 random_configuration_chooser: typing.Union[
                     ChooserNoCoolDown, ChooserLinearCoolDown]=ChooserNoCoolDown(2.0),
                 predict_incumbent: bool=True):
        """
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
        acq_optimizer: AcquisitionFunctionMaximizer
            Optimizer of acquisition function.
        acquisition_function : AcquisitionFunction
            Object that implements the AbstractAcquisitionFunction (i.e., infill
            criterion for acq_optimizer)
        restore_incumbent: Configuration
            incumbent to be used from the start. ONLY used to restore states.
        rng: np.random.RandomState
            Random number generator
        random_configuration_chooser
            Chooser for random configuration -- one of
            * ChooserNoCoolDown(modulus)
            * ChooserLinearCoolDown(start_modulus, modulus_increment, end_modulus)
        predict_incumbent: bool
            Use predicted performance of incumbent instead of observed performance
        """

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)
        self.incumbent = restore_incumbent

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
        self.random_configuration_chooser = random_configuration_chooser

        self._random_search = RandomSearch(
            acquisition_func, self.config_space, rng
        )

        self.predict_incumbent = predict_incumbent

    def start(self):
        """Starts the Bayesian Optimization loop.
        Detects whether we the optimization is restored from previous state.
        """
        self.stats.start_timing()
        # Initialization, depends on input
        if self.stats.ta_runs == 0 and self.incumbent is None:
            self.incumbent = self.initial_design.run()

        elif self.stats.ta_runs > 0 and self.incumbent is None:
            raise ValueError("According to stats there have been runs performed, "
                             "but the optimizer cannot detect an incumbent. Did "
                             "you set the incumbent (e.g. after restoring state)?")
        elif self.stats.ta_runs == 0 and self.incumbent is not None:
            raise ValueError("An incumbent is specified, but there are no runs "
                             "recorded in the Stats-object. If you're restoring "
                             "a state, please provide the Stats-object.")
        else:
            # Restoring state!
            self.logger.info("State Restored! Starting optimization with "
                             "incumbent %s", self.incumbent)
            self.logger.info("State restored with following budget:")
            self.stats.print_stats()

        # To be on the safe side -> never return "None" as incumbent
        if not self.incumbent:
            self.incumbent = self.scenario.cs.get_default_configuration()

    def run(self):
        """Runs the Bayesian optimization loop

        Returns
        ----------
        incumbent: np.array(1, H)
            The best found configuration
        """
        self.start()

        # Main BO loop
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
                            output_directory=self.scenario.output_dir_for_this_run,
                            logger=self.logger)

            logging.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def choose_next(self, X: np.ndarray, Y: np.ndarray,
                    incumbent_value: float=None):
        """Choose next candidate solution with Bayesian optimization. The
        suggested configurations depend on the argument ``acq_optimizer`` to
        the ``SMBO`` class.

        Parameters
        ----------
        X : (N, D) numpy array
            Each row contains a configuration and one set of
            instance features.
        Y : (N, O) numpy array
            The function values for each configuration instance pair.
        incumbent_value: float
            Cost value of incumbent configuration
            (required for acquisition function);
            if not given, it will be inferred from runhistory;
            if not given and runhistory is empty,
            it will raise a ValueError

        Returns
        -------
        Iterable
        """
        if X.shape[0] == 0:
            # Only return a single point to avoid an overly high number of
            # random search iterations
            return self._random_search.maximize(
                runhistory=self.runhistory, stats=self.stats, num_points=1
            )

        self.model.train(X, Y)

        if incumbent_value is None:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent_value = self._get_incumbent_value()

        self.acquisition_func.update(model=self.model, eta=incumbent_value, num_data=len(self.runhistory.data))

        challengers = self.acq_optimizer.maximize(
            runhistory=self.runhistory,
            stats=self.stats,
            num_points=self.scenario.acq_opt_challengers,
            random_configuration_chooser=self.random_configuration_chooser
        )
        return challengers

    def _get_incumbent_value(self):
        ''' get incumbent value either from runhistory
            or from best predicted performance on configs in runhistory
            (depends on self.predict_incumbent)"

            Return
            ------
            float
        '''
        if self.predict_incumbent:
            configs = convert_configurations_to_array(self.runhistory.get_all_configs())
            costs = list(map(
                lambda config:
                    self.model.predict_marginalized_over_instances(config.reshape((1, -1)))[0][0][0],
                configs,
            ))
            incumbent_value = np.min(costs)
            # won't need log(y) if EPM was already trained on log(y)

        else:
            if self.runhistory.empty():
                raise ValueError("Runhistory is empty and the cost value of "
                                 "the incumbent is unknown.")
            incumbent_value = self.runhistory.get_cost(self.incumbent)
            if isinstance(self.rh2EPM,RunHistory2EPM4LogCost):
                incumbent_value = np.log(incumbent_value)

        return incumbent_value

    def validate(self, config_mode='inc', instance_mode='train+test',
                 repetitions=1, use_epm=False, n_jobs=-1, backend='threading'):
        """Create validator-object and run validation, using
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

        Returns
        -------
        runhistory: RunHistory
            runhistory containing all specified runs
        """
        if isinstance(config_mode, str):
            traj_fn = os.path.join(self.scenario.output_dir_for_this_run, "traj_aclib2.json")
            trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=self.scenario.cs)
        else:
            trajectory = None
        if self.scenario.output_dir_for_this_run:
            new_rh_path = os.path.join(self.scenario.output_dir_for_this_run, "validated_runhistory.json")
        else:
            new_rh_path = None

        validator = Validator(self.scenario, trajectory, self.rng)
        if use_epm:
            new_rh = validator.validate_epm(config_mode=config_mode,
                                            instance_mode=instance_mode,
                                            repetitions=repetitions,
                                            runhistory=self.runhistory,
                                            output_fn=new_rh_path)
        else:
            new_rh = validator.validate(config_mode, instance_mode, repetitions,
                                        n_jobs, backend, self.runhistory,
                                        self.intensifier.tae_runner,
                                        output_fn=new_rh_path)
        return new_rh

    def _get_timebound_for_intensification(self, time_spent:float):
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

    def _component_builder(self, conf:typing.Union[Configuration, dict]) \
        -> typing.Tuple[AbstractAcquisitionFunction, AbstractEPM]:
        """
            builds new Acquisition function object
            and EPM object and returns these

            Parameters
            ----------
            conf: typing.Union[Configuration, dict]
                configuration specificing "model" and "acq_func"

            Returns
            -------
            typing.Tuple[AbstractAcquisitionFunction, AbstractEPM]

        """
        types, bounds = get_types(self.config_space, instance_features=self.scenario.feature_array)

        if conf["model"] == "RF":
            model = RandomForestWithInstances(
                  types=types,
                  bounds=bounds,
                  instance_features=self.scenario.feature_array,
                  seed=self.rng.randint(MAXINT),
                  pca_components=conf.get("pca_dim", self.scenario.PCA_DIM),
                  log_y=conf.get("log_y", self.scenario.transform_y in ["LOG", "LOGS"]),
                  num_trees=conf.get("num_trees", self.scenario.rf_num_trees),
                  do_bootstrapping=conf.get("do_bootstrapping", self.scenario.rf_do_bootstrapping),
                  ratio_features=conf.get("ratio_features", self.scenario.rf_ratio_features),
                  min_samples_split=conf.get("min_samples_split", self.scenario.rf_min_samples_split),
                  min_samples_leaf=conf.get("min_samples_leaf", self.scenario.rf_min_samples_leaf),
                  max_depth=conf.get("max_depth", self.scenario.rf_max_depth))

        elif conf["model"] == "GP":
            cov_amp = 2
            n_dims = len(types)

            initial_ls = np.ones([n_dims])
            exp_kernel = george.kernels.Matern52Kernel(initial_ls, ndim=n_dims)
            kernel = cov_amp * exp_kernel

            prior = DefaultPrior(len(kernel) + 1, rng=self.rng)

            n_hypers = 3 * len(kernel)
            if n_hypers % 2 == 1:
                n_hypers += 1

            model = GaussianProcessMCMC(
                types=types,
                bounds=bounds,
                kernel=kernel,
                prior=prior,
                n_hypers=n_hypers,
                chain_length=200,
                burnin_steps=100,
                normalize_input=True,
                normalize_output=True,
                seed=self.rng.randint(low=0, high=10000),
            )

        if conf["acq_func"] == "EI":
            acq = EI(model=model,
                     par=conf.get("par_ei", 0))
        elif conf["acq_func"] == "LCB":
            acq = LCB(model=model,
                par=conf.get("par_lcb", 0))
        elif conf["acq_func"] == "PI":
            acq = PI(model=model,
                     par=conf.get("par_pi", 0))
        elif conf["acq_func"] == "LogEI":
            # par value should be in log-space
            acq = LogEI(model=model,
                        par=conf.get("par_logei", 0))

        return acq, model

    def _get_acm_cs(self):
        """
            returns a configuration space
            designed for querying ~smac.optimizer.smbo._component_builder

            Returns
            -------
                ConfigurationSpace
        """

        cs = ConfigurationSpace()
        cs.seed(self.rng.randint(0,2**20))

        model = CategoricalHyperparameter("model", choices=("RF", "GP"))

        num_trees = Constant("num_trees", value=10)
        bootstrap = CategoricalHyperparameter("do_bootstrapping", choices=(True, False), default_value=True)
        ratio_features = CategoricalHyperparameter("ratio_features", choices=(3 / 6, 4 / 6, 5 / 6, 1), default_value=1)
        min_split = UniformIntegerHyperparameter("min_samples_to_split", lower=1, upper=10, default_value=2)
        min_leaves = UniformIntegerHyperparameter("min_samples_in_leaf", lower=1, upper=10, default_value=1)

        cs.add_hyperparameters([model, num_trees, bootstrap, ratio_features, min_split, min_leaves])

        inc_num_trees = InCondition(num_trees, model, ["RF"])
        inc_bootstrap = InCondition(bootstrap, model, ["RF"])
        inc_ratio_features = InCondition(ratio_features, model, ["RF"])
        inc_min_split = InCondition(min_split, model, ["RF"])
        inc_min_leavs = InCondition(min_leaves, model, ["RF"])

        cs.add_conditions([inc_num_trees, inc_bootstrap, inc_ratio_features, inc_min_split, inc_min_leavs])

        acq  = CategoricalHyperparameter("acq_func", choices=("EI", "LCB", "PI", "LogEI"))
        par_ei = UniformFloatHyperparameter("par_ei", lower=-10, upper=10)
        par_pi = UniformFloatHyperparameter("par_pi", lower=-10, upper=10)
        par_logei = UniformFloatHyperparameter("par_logei", lower=0.001, upper=100, log=True)
        par_lcb = UniformFloatHyperparameter("par_lcb", lower=0.0001, upper=0.9999)

        cs.add_hyperparameters([acq, par_ei, par_pi, par_logei, par_lcb])

        inc_par_ei = InCondition(par_ei, acq, ["EI"])
        inc_par_pi = InCondition(par_pi, acq, ["PI"])
        inc_par_logei = InCondition(par_logei, acq, ["LogEI"])
        inc_par_lcb = InCondition(par_lcb, acq, ["LCB"])

        cs.add_conditions([inc_par_ei, inc_par_pi, inc_par_logei, inc_par_lcb])

        return cs
