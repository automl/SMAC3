import os
import logging
import numpy as np
import time
import typing

import smac
from smac.configspace import ConfigurationSpace, Configuration, Constant,\
     CategoricalHyperparameter, UniformFloatHyperparameter, \
     UniformIntegerHyperparameter, InCondition
from smac.epm.base_epm import AbstractEPM
from smac.epm.rf_with_instances import RandomForestWithInstances
from smac.epm.gaussian_process_mcmc import GaussianProcessMCMC
from smac.epm.gp_base_prior import LognormalPrior, HorseshoePrior
from smac.epm.util_funcs import get_types
from smac.initial_design.initial_design import InitialDesign
from smac.intensification.abstract_racer import AbstractRacer
from smac.optimizer import pSMAC
from smac.optimizer.acquisition import AbstractAcquisitionFunction, EI, LogEI,\
    LCB, PI
from smac.optimizer.random_configuration_chooser import ChooserNoCoolDown, \
    ChooserLinearCoolDown
from smac.optimizer.ei_optimization import AcquisitionFunctionMaximizer
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.runhistory.runhistory import RunHistory
from smac.runhistory.runhistory2epm import AbstractRunHistory2EPM
from smac.scenario.scenario import Scenario
from smac.stats.stats import Stats
from smac.utils.io.traj_logging import TrajLogger
from smac.utils.validate import Validator
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
    intensifier
    aggregate_func
    num_run
    rng
    initial_design_configs
    epm_chooser
    """

    def __init__(self,
                 scenario: Scenario,
                 stats: Stats,
                 initial_design: InitialDesign,
                 runhistory: RunHistory,
                 runhistory2epm: AbstractRunHistory2EPM,
                 intensifier: AbstractRacer,
                 aggregate_func: callable,
                 num_run: int,
                 model: RandomForestWithInstances,
                 acq_optimizer: AcquisitionFunctionMaximizer,
                 acquisition_func: AbstractAcquisitionFunction,
                 rng: np.random.RandomState,
                 restore_incumbent: Configuration = None,
                 random_configuration_chooser: typing.Union[
                     ChooserNoCoolDown, ChooserLinearCoolDown]=ChooserNoCoolDown(2.0),
                 predict_incumbent: bool = True):
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
        acquisition_func : AcquisitionFunction
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
        self.intensifier = intensifier
        self.aggregate_func = aggregate_func
        self.num_run = num_run
        self.rng = rng

        self.initial_design_configs = []

        # initialize the chooser to get configurations from the EPM
        self.epm_chooser = EPMChooser(scenario=scenario,
                                      stats=stats,
                                      runhistory=runhistory,
                                      runhistory2epm=runhistory2epm,
                                      model=model,
                                      acq_optimizer=acq_optimizer,
                                      acquisition_func=acquisition_func,
                                      rng=rng,
                                      restore_incumbent=restore_incumbent,
                                      random_configuration_chooser=random_configuration_chooser,
                                      predict_incumbent=predict_incumbent)

    def start(self):
        """Starts the Bayesian Optimization loop.
        Detects whether we the optimization is restored from previous state.
        """
        self.stats.start_timing()

        # Initialization, depends on input
        if self.stats.ta_runs == 0 and self.incumbent is None:
            self.logger.info('Running initial design')
            # Intensifier initialization
            self.initial_design_configs = self.initial_design.select_configurations()

            # to be on the safe side, never return an empty list of initial configs
            if not self.initial_design_configs:
                self.initial_design_configs = [self.config_space.get_default_configuration()]

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

            # sample next configuration for intensification
            # Initial design runs are also included in the BO loop now.
            challenger, new_challenger = self.intensifier.get_next_challenger(
                challengers=self.initial_design_configs,
                chooser=self.epm_chooser,
                run_history=self.runhistory,
                repeat_configs=self.intensifier.repeat_configs
            )

            # remove config from initial design challengers to not repeat it again
            self.initial_design_configs = [c for c in self.initial_design_configs if c != challenger]

            # update timebound only if a 'new' configuration is sampled as the challenger
            if new_challenger:
                time_spent = time.time() - start_time
                time_left = self._get_timebound_for_intensification(time_spent)

            if challenger:
                # evaluate selected challenger
                self.logger.debug("Intensify - evaluate challenger")

                self.incumbent, inc_perf = self.intensifier.eval_challenger(
                    challenger=challenger,
                    incumbent=self.incumbent,
                    run_history=self.runhistory,
                    aggregate_func=self.aggregate_func,
                    time_bound=max(self.intensifier._min_time, time_left))

                if self.scenario.shared_model:
                    pSMAC.write(run_history=self.runhistory,
                                output_directory=self.scenario.output_dir_for_this_run,
                                logger=self.logger)

            self.logger.debug("Remaining budget: %f (wallclock), %f (ta costs), %f (target runs)" % (
                self.stats.get_remaing_time_budget(),
                self.stats.get_remaining_ta_budget(),
                self.stats.get_remaining_ta_runs()))

            if self.stats.is_budget_exhausted():
                break

            self.stats.print_stats(debug_out=True)

        return self.incumbent

    def validate(self,
                 config_mode='inc',
                 instance_mode='train+test',
                 repetitions=1,
                 use_epm=False,
                 n_jobs=-1,
                 backend='threading'):
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
            trajectory = TrajLogger.read_traj_aclib_format(fn=traj_fn, cs=self.config_space)
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

    def _get_timebound_for_intensification(self, time_spent: float):
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
                             frac_intensify)
        total_time = time_spent / (1 - frac_intensify)
        time_left = frac_intensify * total_time
        self.logger.debug("Total time: %.4f, time spent on choosing next "
                          "configurations: %.4f (%.2f), time left for "
                          "intensification: %.4f (%.2f)" %
                          (total_time, time_spent, (1 - frac_intensify), time_left, frac_intensify))
        return time_left

    def _component_builder(self, conf: typing.Union[Configuration, dict]) \
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
                configspace=self.config_space,
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
                max_depth=conf.get("max_depth", self.scenario.rf_max_depth),
            )

        elif conf["model"] == "GP":
            from smac.epm.gp_kernels import ConstantKernel, HammingKernel, WhiteKernel, Matern

            cov_amp = ConstantKernel(
                2.0,
                constant_value_bounds=(np.exp(-10), np.exp(2)),
                prior=LognormalPrior(mean=0.0, sigma=1.0, rng=self.rng),
            )

            cont_dims = np.nonzero(types == 0)[0]
            cat_dims = np.nonzero(types != 0)[0]

            if len(cont_dims) > 0:
                exp_kernel = Matern(
                    np.ones([len(cont_dims)]),
                    [(np.exp(-10), np.exp(2)) for _ in range(len(cont_dims))],
                    nu=2.5,
                    operate_on=cont_dims,
                )

            if len(cat_dims) > 0:
                ham_kernel = HammingKernel(
                    np.ones([len(cat_dims)]),
                    [(np.exp(-10), np.exp(2)) for _ in range(len(cat_dims))],
                    operate_on=cat_dims,
                )
            noise_kernel = WhiteKernel(
                noise_level=1e-8,
                noise_level_bounds=(np.exp(-25), np.exp(2)),
                prior=HorseshoePrior(scale=0.1, rng=self.rng),
            )

            if len(cont_dims) > 0 and len(cat_dims) > 0:
                # both
                kernel = cov_amp * (exp_kernel * ham_kernel) + noise_kernel
            elif len(cont_dims) > 0 and len(cat_dims) == 0:
                # only cont
                kernel = cov_amp * exp_kernel + noise_kernel
            elif len(cont_dims) == 0 and len(cat_dims) > 0:
                # only cont
                kernel = cov_amp * ham_kernel + noise_kernel
            else:
                raise ValueError()

            n_mcmc_walkers = 3 * len(kernel.theta)
            if n_mcmc_walkers % 2 == 1:
                n_mcmc_walkers += 1

            model = GaussianProcessMCMC(
                self.config_space,
                types=types,
                bounds=bounds,
                kernel=kernel,
                n_mcmc_walkers=n_mcmc_walkers,
                chain_length=250,
                burnin_steps=250,
                normalize_y=True,
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

        if 'gp' in smac.extras_installed:
            model = CategoricalHyperparameter("model", choices=("RF", "GP"))
        else:
            model = Constant("model", value="RF")

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
