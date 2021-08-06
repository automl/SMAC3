import logging
import typing
import warnings

import numpy as np

from smac.intensification.abstract_racer import AbstractRacer, RunInfoIntent
from smac.optimizer.epm_configuration_chooser import EPMChooser
from smac.intensification.parallel_scheduling import ParallelScheduler
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, RunInfo, RunValue
from smac.tae import StatusType
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2019, ML4AAD"
__license__ = "3-clause BSD"


class _SuccessiveHalving(AbstractRacer):
    """ Races multiple challengers against an incumbent using Successive Halving method

    This class contains the logic to implement:
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    Supplementary reference: http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf

    The `SuccessiveHalving` class can create multiple `_SuccessiveHalving` objects, to
    allow parallelism in the method (up to the number of workers available). The user  interface
    is expected to be `SuccessiveHalving`, yet this class (`_SuccessiveHalving`) contains the
    actual single worker implementation of the SMAC4MF method.

    Successive Halving intensifier (and Hyperband) can operate on two kinds of budgets:

    1. **'Instances' as budget**:

       When multiple instances are provided or when run objective is "runtime",
       this is the criterion used as budget
       for successive halving iterations i.e., the budget determines how many
       instances the challengers are evaluated
       on at a time. Top challengers for the next iteration are selected based
       on the combined performance across all instances used.

       If ``initial_budget`` and ``max_budget`` are not provided, then they are
       set to 1 and total number of available instances respectively by default.

    2. **'Real-valued' budget**:

       This is used when there is only one instance provided and when run
       objective is "quality", i.e. budget is a positive, real-valued number
       that can be passed to the target algorithm as an argument.
       It can be used to control anything by the target algorithm,
       Eg: number of epochs for training a neural network.

       ``initial_budget`` and ``max_budget`` are required parameters for
       this type of budget.

    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 _all_budgets: typing.Optional[typing.List[float]] = None,
                 _n_configs_in_stage: typing.Optional[typing.List[int]] = None,
                 num_initial_challengers: typing.Optional[int] = None,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order: typing.Optional[str] = 'shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 inst_seed_pairs: typing.Optional[typing.List[typing.Tuple[str, int]]] = None,
                 min_chall: int = 1,
                 incumbent_selection: str = 'highest_executed_budget',
                 identifier: int = 0,
                 ) -> None:
        """
        Parameters
        ----------
        stats: smac.stats.stats.Stats
            stats object
        traj_logger: smac.utils.io.traj_logging.TrajLogger
            TrajLogger object to log all new incumbents
        rng : np.random.RandomState
        instances : typing.List[str]
            list of all instance ids
        instance_specifics : typing.Mapping[str,np.ndarray]
            mapping from instance name to instance specific string
        cutoff : typing.Optional[int]
            cutoff of TA runs
        deterministic : bool
            whether the TA is deterministic or not
        initial_budget : typing.Optional[float]
            minimum budget allowed for 1 run of successive halving
        max_budget : typing.Optional[float]
            maximum budget allowed for 1 run of successive halving
        eta : float
            'halving' factor after each iteration in a successive halving run. Defaults to 3
        _all_budgets: typing.Optional[typing.List[float]] = None
            Used internally when HB uses SH as a subrouting
        _n_configs_in_stage: typing.Optional[typing.List[int]] = None
            Used internally when HB uses SH as a subrouting
        num_initial_challengers : typing.Optional[int]
            number of challengers to consider for the initial budget. If None, calculated internally
        run_obj_time : bool
            whether the run objective is runtime or not (if true, apply adaptive capping)
        n_seeds : typing.Optional[int]
            Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
        instance_order : typing.Optional[str]
            how to order instances. Can be set to: [None, shuffle_once, shuffle]

            * None - use as is given by the user
            * shuffle_once - shuffle once and use across all SH run (default)
            * shuffle - shuffle before every SH run

        adaptive_capping_slackfactor : float
            slack factor of adpative capping (factor * adaptive cutoff)
        inst_seed_pairs : typing.List[typing.Tuple[str, int]], optional
            Do not set this argument, it will only be used by hyperband!
        min_chall: int
            minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
            raise an exception if a value larger than 1 is passed.
        incumbent_selection: str
            How to select incumbent in successive halving. Only active for real-valued budgets.
            Can be set to: [highest_executed_budget, highest_budget, any_budget]

            * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
            * highest_budget - incumbent is selected only based on the highest budget
            * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
        identifier: int
            Adds a numerical identifier on this SH instance. Used for debug and tagging
            logger messages properly
        """

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall)

        self.identifier = identifier
        self.logger = logging.getLogger(
            self.__module__ + "." + str(self.identifier) + "." + self.__class__.__name__)

        if self.min_chall > 1:
            raise ValueError('Successive Halving cannot handle argument `min_chall` > 1.')
        self.first_run = True

        # INSTANCES
        self.n_seeds = n_seeds if n_seeds else 1
        self.instance_order = instance_order

        # NOTE Remove after solving how to handle multiple seeds and 1 instance
        if len(self.instances) == 1 and self.n_seeds > 1:
            raise NotImplementedError('This case (multiple seeds and 1 instance) cannot be handled yet!')

        # if instances are coming from Hyperband, skip the instance preprocessing section
        # it is already taken care by Hyperband

        if not inst_seed_pairs:
            # set seed(s) for all SH runs
            # - currently user gives the number of seeds to consider
            if self.deterministic:
                seeds = [0]
            else:
                seeds = [int(s) for s in self.rs.randint(low=0, high=MAXINT, size=self.n_seeds)]
                if self.n_seeds == 1:
                    self.logger.warning('The target algorithm is specified to be non deterministic, '
                                        'but number of seeds to evaluate are set to 1. '
                                        'Consider setting `n_seeds` > 1.')

            # storing instances & seeds as tuples
            self.inst_seed_pairs = [(i, s) for s in seeds for i in self.instances]

            # determine instance-seed pair order
            if self.instance_order == 'shuffle_once':
                # randomize once
                self.rs.shuffle(self.inst_seed_pairs)
        else:
            self.inst_seed_pairs = inst_seed_pairs

        # successive halving parameters
        self._init_sh_params(initial_budget=initial_budget, max_budget=max_budget, eta=eta,
                             num_initial_challengers=num_initial_challengers,
                             _all_budgets=_all_budgets, _n_configs_in_stage=_n_configs_in_stage)

        # adaptive capping
        if self.instance_as_budget and self.instance_order != 'shuffle' and self.run_obj_time:
            self.adaptive_capping = True
        else:
            self.adaptive_capping = False

        # challengers can be repeated only if optimizing across multiple seeds or changing instance orders every run
        # (this does not include having multiple instances)
        if self.n_seeds > 1 or self.instance_order == 'shuffle':
            self.repeat_configs = True
        else:
            self.repeat_configs = False

        # incumbent selection design
        assert incumbent_selection in ['highest_executed_budget', 'highest_budget', 'any_budget']
        self.incumbent_selection = incumbent_selection

        # Define state variables to please mypy

        # current instance index tracks two things. Configurations that are to be launched,
        # That is config A needs to run in 3 instances/seed pairs, then curr_inst_idx should
        # track this. But then, if a new configuration is added in the case of parallelism
        # a new separate curr_inst_idx needs to be started
        self.curr_inst_idx = {}  # type: typing.Dict[Configuration, int]
        self.running_challenger = None
        self.success_challengers = set()  # type: typing.Set[Configuration]
        self.do_not_advance_challengers = set()  # type: typing.Set[Configuration]
        self.fail_challengers = set()  # type: typing.Set[Configuration]
        self.fail_chal_offset = 0

        # Track which configs were launched. This will allow to have an extra check to make sure
        # that a successive halver deals only with the configs it launched,
        # but also allows querying the status of the configs via the run history.
        # In other works, the run history is agnostic of the origin of the configurations,
        # that is, which successive halving instance created it. The RunInfo object
        # is aware of this information, and for parallel execution, the routing of
        # finish results is expected to use this information.
        # Nevertheless, the common object among SMBO/intensifier, which is the
        # run history, does not have this information and so we track locally. That way,
        # when we access the complete list of configs from the run history, we filter
        # the ones launched by the current succesive halver using self.run_tracker
        self.run_tracker = {}  # type: typing.Dict[typing.Tuple[Configuration, str, int, float], bool]

    def _init_sh_params(self,
                        initial_budget: typing.Optional[float],
                        max_budget: typing.Optional[float],
                        eta: float,
                        num_initial_challengers: typing.Optional[int] = None,
                        _all_budgets: typing.Optional[typing.List[float]] = None,
                        _n_configs_in_stage: typing.Optional[typing.List[int]] = None,
                        ) -> None:
        """ Initialize Successive Halving parameters

        Parameters
        ----------
        initial_budget : typing.Optional[float]
            minimum budget allowed for 1 run of successive halving
        max_budget : typing.Optional[float]
            maximum budget allowed for 1 run of successive halving
        eta : float
            'halving' factor after each iteration in a successive halving run
        num_initial_challengers : typing.Optional[int]
            number of challengers to consider for the initial budget
        _all_budgets: typing.Optional[typing.List[float]] = None
            Used internally when HB uses SH as a subrouting
        _n_configs_in_stage: typing.Optional[typing.List[int]] = None
            Used internally when HB uses SH as a subrouting
        """

        if eta <= 1:
            raise ValueError('eta must be greater than 1')
        self.eta = eta

        # BUDGETS

        if max_budget is not None and initial_budget is not None \
                and max_budget < initial_budget:
            raise ValueError('Max budget has to be larger than min budget')

        # - if only 1 instance was provided & quality objective, then use cutoff as budget
        # - else, use instances as budget
        if not self.run_obj_time and len(self.inst_seed_pairs) <= 1:
            # budget with cutoff
            if initial_budget is None or max_budget is None:
                raise ValueError("Successive Halving with real-valued budget (i.e., only 1 instance) "
                                 "requires parameters initial_budget and max_budget for intensification!")

            self.initial_budget = initial_budget
            self.max_budget = max_budget
            self.instance_as_budget = False

        else:
            # budget with instances
            if self.run_obj_time and len(self.inst_seed_pairs) <= 1:
                self.logger.warning("Successive Halving has objective 'runtime' but only 1 instance-seed pair.")
            self.initial_budget = int(initial_budget) if initial_budget else 1
            self.max_budget = int(max_budget) if max_budget else len(self.inst_seed_pairs)
            self.instance_as_budget = True

            if self.max_budget > len(self.inst_seed_pairs):
                raise ValueError('Max budget cannot be greater than the number of instance-seed pairs')
            if self.max_budget < len(self.inst_seed_pairs):
                self.logger.warning('Max budget (%d) does not include all instance-seed pairs (%d)' %
                                    (self.max_budget, len(self.inst_seed_pairs)))

        budget_type = 'INSTANCES' if self.instance_as_budget else 'REAL-VALUED'
        self.logger.info("Successive Halving configuration: budget type = %s, "
                         "Initial budget = %.2f, Max. budget = %.2f, eta = %.2f" %
                         (budget_type, self.initial_budget, self.max_budget, self.eta))

        # precomputing stuff for SH
        # max. no. of SH iterations possible given the budgets
        max_sh_iter = int(np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.eta)))
        # initial number of challengers to sample
        if num_initial_challengers is None:
            num_initial_challengers = int(self.eta ** max_sh_iter)

        if _all_budgets is not None and _n_configs_in_stage is not None:
            # Assert we use the given numbers to avoid rounding issues, see #701
            self.all_budgets = _all_budgets
            self.n_configs_in_stage = _n_configs_in_stage
        else:
            # budgets to consider in each stage
            self.all_budgets = self.max_budget * np.power(self.eta, -np.linspace(max_sh_iter, 0,
                                                                                 max_sh_iter + 1))
            # number of challengers to consider in each stage
            n_configs_in_stage = num_initial_challengers * \
                np.power(self.eta, -np.linspace(0, max_sh_iter, max_sh_iter + 1))
            self.n_configs_in_stage = np.array(np.round(n_configs_in_stage), dtype=int).tolist()

    def process_results(self,
                        run_info: RunInfo,
                        incumbent: typing.Optional[Configuration],
                        run_history: RunHistory,
                        time_bound: float,
                        result: RunValue,
                        log_traj: bool = True,
                        ) -> \
            typing.Tuple[Configuration, float]:
        """
        The intensifier stage will be updated based on the results/status
        of a configuration execution.
        Also, a incumbent will be determined.

        Parameters
        ----------
        run_info : RunInfo
               A RunInfo containing the configuration that was evaluated
        incumbent : typing.Optional[Configuration]
            Best configuration seen so far
        run_history : RunHistory
            stores all runs we ran so far
            if False, an evaluated configuration will not be generated again
        time_bound : float
            time in [sec] available to perform intensify
        result: RunValue
            Contain the result (status and other methadata) of exercising
            a challenger/incumbent.
        log_traj: bool
            Whether to log changes of incumbents in trajectory

        Returns
        -------
        incumbent: Configuration
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """

        # Mark the fact that we processed this configuration
        self.run_tracker[(run_info.config, run_info.instance, run_info.seed, run_info.budget)] = True

        # If The incumbent is None and it is the first run, we use the challenger
        if not incumbent and self.first_run:
            self.logger.info(
                "First run, no incumbent provided; challenger is assumed to be the incumbent"
            )
            incumbent = run_info.config
            self.first_run = False

        # In a serial run, if we have to CAP a run, then we stop launching
        # more configurations for this run.
        # In the context of parallelism, we launch the instances proactively
        # The fact that self.curr_inst_idx[run_info.config] is np.inf means
        # that no more instances will be launched for the current config, so we
        # can add a check to make sure that if we are capping, this makes sense
        # for the active challenger
        if result.status == StatusType.CAPPED and run_info.config == self.running_challenger:
            self.curr_inst_idx[run_info.config] = np.inf
        else:
            self._ta_time = self._ta_time  # type: float # make mypy happy
            self.num_run = self.num_run  # type: int # make mypy happy
            self._ta_time += result.time
            self.num_run += 1

        # 0: Before moving to a new stage, we have to complete M x N tasks, where M is the
        # total number of configurations evaluated in N instance/seed pairs.
        # The last active configuration is stored in self.running challengers is the M
        # configuration, and to get to this point, we have already submitted tasks
        # for (M - 1) configurations on N instances seed-pairs. The status of such
        # (M - 1) * N tasks is tracked in self.run_tracker, that has a value of False
        # if not processed and true if such task has been processed.
        # This stage is complete only if all tasks have been launched and all of the
        # already launched tasks are processed.

        # 1: We first query if we have launched everything already (All M * N tasks)
        all_config_inst_seed_launched = self._all_config_inst_seed_pairs_launched(
            run_history=run_history,
            activate_configuration_being_intensified=self.running_challenger,
        )

        # 2: Then we get the already submitted tasks (that is, proposed by get_next_runs),
        # that have not yet been processed process_results
        all_config_inst_seeds_processed = len([v for v in self.run_tracker.values() if not v]) <= 0

        # 3: Then the total number of remaining task before we can conclude this stage
        # is calculated by taking into account point 2 and 3 above
        is_stage_done = all_config_inst_seed_launched and all_config_inst_seeds_processed

        # adding challengers to the list of evaluated challengers
        #  - Stop: CAPPED/CRASHED/TIMEOUT/MEMOUT/DONOTADVANCE (!= SUCCESS)
        #  - Advance to next stage: SUCCESS
        # curr_challengers is a set, so "at least 1" success can be counted by set addition (no duplicates)
        # If a configuration is successful, it is added to curr_challengers.
        # if it fails it is added to fail_challengers.
        if np.isfinite(self.curr_inst_idx[run_info.config]) and result.status == StatusType.SUCCESS:
            self.success_challengers.add(run_info.config)  # successful configs
        elif np.isfinite(self.curr_inst_idx[run_info.config]) and result.status == StatusType.DONOTADVANCE:
            self.do_not_advance_challengers.add(run_info.config)
        else:
            self.fail_challengers.add(run_info.config)  # capped/crashed/do not advance configs

        # We need to update the incumbent if this config we are processing
        # completes all scheduled instance-seed pairs.
        # Here, a config/seed/instance is going to be processed for the first time
        # (it has been previously scheduled by get_next_run and marked False, indicating
        # that it has not been processed yet. Entering process_results() this config/seed/instance
        # is marked as TRUE as an indication that it has finished and should be processed)
        # so if all configurations runs are marked as TRUE it means that this new config
        # was the missing piece to have everything needed to compare against the incumbent
        update_incumbent = all([v for k, v in self.run_tracker.items() if k[0] == run_info.config])

        # get incumbent if all instances have been evaluated
        if is_stage_done or update_incumbent:
            incumbent = self._compare_configs(challenger=run_info.config,
                                              incumbent=incumbent,
                                              run_history=run_history,
                                              log_traj=log_traj)
        if is_stage_done:
            self.logger.info('Successive Halving iteration-step: %d-%d with '
                             'budget [%.2f / %d] - evaluated %d challenger(s)' %
                             (self.sh_iters + 1, self.stage + 1, self.all_budgets[self.stage], self.max_budget,
                              self.n_configs_in_stage[self.stage]))

            self._update_stage(run_history=run_history)

        # get incumbent cost
        inc_perf = run_history.get_cost(incumbent)

        return incumbent, inc_perf

    def get_next_run(self,
                     challengers: typing.Optional[typing.List[Configuration]],
                     incumbent: Configuration,
                     chooser: typing.Optional[EPMChooser],
                     run_history: RunHistory,
                     repeat_configs: bool = True,
                     num_workers: int = 1,
                     ) -> typing.Tuple[RunInfoIntent, RunInfo]:
        """
        Selects which challenger to use based on the iteration stage and set the iteration parameters.
        First iteration will choose configurations from the ``chooser`` or input challengers,
        while the later iterations pick top configurations from the previously selected challengers in that iteration

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent: Configuration
            incumbent configuration
        chooser : smac.optimizer.epm_configuration_chooser.EPMChooser
            optimizer that generates next configurations to use for racing
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        repeat_configs : bool
            if False, an evaluated configuration will not be generated again
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
        intent: RunInfoIntent
               Indicator of how to consume the RunInfo object
        run_info: RunInfo
            An object that encapsulates the minimum information to
            evaluate a configuration
         """
        if num_workers > 1:
            warnings.warn("Consider using ParallelSuccesiveHalving instead of "
                          "SuccesiveHalving. The later will halt on each stage "
                          "transition until all configs for the current stage are completed."
                          )
        # if this is the first run, then initialize tracking variables
        if not hasattr(self, 'stage'):
            self._update_stage(run_history=run_history)

        # In the case of multiprocessing, we have runs in Running stage, which have not
        # been processed via process_results(). get_next_run() is called agnostically by
        # smbo. To prevent launching more configs, than the ones needed, we query if
        # there is room for more configurations, else we wait for process_results()
        # to trigger a new stage
        if self._all_config_inst_seed_pairs_launched(run_history, self.running_challenger):
            return RunInfoIntent.WAIT, RunInfo(
                config=None,
                instance=None,
                instance_specific="0",
                seed=0,
                cutoff=self.cutoff,
                capped=False,
                budget=0.0,
                source_id=self.identifier,
            )

        # sampling from next challenger marks the beginning of a new iteration
        self.iteration_done = False

        curr_budget = self.all_budgets[self.stage]

        # if all instances have been executed, then reset and move on to next config
        if self.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            n_insts = (int(curr_budget) - prev_budget)
        else:
            n_insts = len(self.inst_seed_pairs)

        # The instances remaining tell us, per configuration, how many instances we
        # have suggested to SMBO
        n_insts_remaining = n_insts
        if self.running_challenger is not None:
            n_insts_remaining = n_insts - self.curr_inst_idx[self.running_challenger]

        # if there are instances pending, finish running configuration
        if self.running_challenger and n_insts_remaining > 0:
            challenger = self.running_challenger
            new_challenger = False
        else:
            # select next configuration
            if self.stage == 0:
                # first stage, so sample from configurations/chooser provided
                challenger = self._next_challenger(challengers=challengers,
                                                   chooser=chooser,
                                                   run_history=run_history,
                                                   repeat_configs=repeat_configs)
                if challenger is None:
                    # If no challenger was sampled from the EPM or
                    # initial challengers, it might mean that the EPM
                    # is proposing a configuration that is currently running.
                    # There is a filtering on the above _next_challenger to return
                    # None if the proposed config us already in the run history
                    # To get a new config, we wait for more data
                    return RunInfoIntent.WAIT, RunInfo(
                        config=None,
                        instance=None,
                        instance_specific="0",
                        seed=0,
                        cutoff=self.cutoff,
                        capped=False,
                        budget=0.0,
                        source_id=self.identifier,
                    )

                new_challenger = True
            else:
                # sample top configs from previously sampled configurations
                try:
                    challenger = self.configs_to_run.pop(0)
                    new_challenger = False
                except IndexError:
                    # self.configs_to_run is populated via update_stage,
                    # which is triggered after the completion of a run
                    # If by there are no more configs to run (which is the case
                    # if we run into a IndexError),
                    return RunInfoIntent.SKIP, RunInfo(
                        config=None,
                        instance=None,
                        instance_specific="0",
                        seed=0,
                        cutoff=self.cutoff,
                        capped=False,
                        budget=0.0,
                        source_id=self.identifier,
                    )

            if challenger:
                # We see a challenger for the first time, so no
                # instance has been launched
                self.curr_inst_idx[challenger] = 0
                self._chall_indx = self._chall_indx  # type: int # make mypy happy
                self._chall_indx += 1
                self.running_challenger = challenger

        # calculating the incumbent's performance for adaptive capping
        # this check is required because:
        #   - there is no incumbent performance for the first ever 'intensify' run (from initial design)
        #   - during the 1st intensify run, the incumbent shouldn't be capped after being compared against itself
        if incumbent and incumbent != challenger:
            inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
            inc_sum_cost = run_history.sum_cost(config=incumbent, instance_seed_budget_keys=inc_runs)
        else:
            inc_sum_cost = np.inf
            if self.first_run:
                self.logger.info("First run, no incumbent provided; challenger is assumed to be the incumbent")
                incumbent = challenger

        # selecting instance-seed subset for this budget, depending on the kind of budget
        if self.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            curr_insts = self.inst_seed_pairs[int(prev_budget):int(curr_budget)]
        else:
            curr_insts = self.inst_seed_pairs

        self.logger.debug(" Running challenger  -  %s" % str(challenger))

        # run the next instance-seed pair for the given configuration
        instance, seed = curr_insts[self.curr_inst_idx[challenger]]

        # selecting cutoff if running adaptive capping
        cutoff = self.cutoff
        if self.run_obj_time:
            cutoff = self._adapt_cutoff(challenger=challenger,
                                        run_history=run_history,
                                        inc_sum_cost=inc_sum_cost)
            if cutoff is not None and cutoff <= 0:
                # ran out of time to validate challenger
                self.logger.debug("Stop challenger intensification due to adaptive capping.")
                self.curr_inst_idx[challenger] = np.inf

        self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

        # For testing purposes, this attribute highlights whether a
        # new challenger is proposed or not. Not required from a functional
        # perspective
        self.new_challenger = new_challenger

        capped = False
        if (self.cutoff is not None) and (cutoff < self.cutoff):  # type: ignore[operator] # noqa F821
            capped = True

        budget = 0.0 if self.instance_as_budget else curr_budget

        self.run_tracker[(challenger, instance, seed, budget)] = False

        # self.curr_inst_idx Tell us our current instance to be run. The upcoming return
        # will launch a challenger on a given instance/seed/pair. The next time this function
        # is called, we will like to run self.curr_inst_idx + 1 for this configuration
        self.curr_inst_idx[challenger] += 1

        return RunInfoIntent.RUN, RunInfo(
            config=challenger,
            instance=instance,
            instance_specific=self.instance_specifics.get(instance, "0"),
            seed=seed,
            cutoff=cutoff,
            capped=capped,
            budget=budget,
            source_id=self.identifier,
        )

    def _update_stage(self, run_history: RunHistory) -> None:
        """
        Update tracking information for a new stage/iteration and update statistics.
        This method is called to initialize stage variables and after all configurations
        of a successive halving stage are completed.

        Parameters
        ----------
         run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        """

        if not hasattr(self, 'stage'):
            # initialize all relevant variables for first run
            # (this initialization is not a part of init because hyperband uses the same init method and has a )
            # to track iteration and stage
            self.sh_iters = 0
            self.stage = 0
            # to track challengers across stages
            self.configs_to_run = []  # type: typing.List[Configuration]
            self.curr_inst_idx = {}
            self.running_challenger = None
            self.success_challengers = set()  # successful configs
            self.do_not_advance_challengers = set()  # configs which are successful, but should not be advanced
            self.fail_challengers = set()  # capped configs and other failures
            self.fail_chal_offset = 0

        else:
            self.stage += 1
            # only uncapped challengers are considered valid for the next iteration
            valid_challengers = list(
                (self.success_challengers | self.do_not_advance_challengers) - self.fail_challengers
            )

            if self.stage < len(self.all_budgets) and len(valid_challengers) > 0:
                # if this is the next stage in same iteration,
                # use top 'k' from the evaluated configurations for next iteration

                # determine 'k' for the next iteration - at least 1
                next_n_chal = int(max(1, self.n_configs_in_stage[self.stage]))
                # selecting the top 'k' challengers for the next iteration
                configs_to_run = self._top_k(configs=valid_challengers,
                                             run_history=run_history,
                                             k=next_n_chal)
                self.configs_to_run = [
                    config for config in configs_to_run
                    if config not in self.do_not_advance_challengers
                ]
                # if some runs were capped, top_k returns less than the required configurations
                # to handle that, we keep track of how many configurations are missing
                # (since they are technically failed here too)
                missing_challengers = int(self.n_configs_in_stage[self.stage]) - len(self.configs_to_run)
                if missing_challengers > 0:
                    self.fail_chal_offset = missing_challengers
                else:
                    self.fail_chal_offset = 0
                if next_n_chal == missing_challengers:
                    next_stage = True
                    self.logger.info('Successive Halving iteration-step: %d-%d with '
                                     'budget [%.2f / %d] - expected %d new challenger(s), but '
                                     'no configurations propagated to the next budget.',
                                     self.sh_iters + 1, self.stage + 1, self.all_budgets[self.stage],
                                     self.max_budget, self.n_configs_in_stage[self.stage])
                else:
                    next_stage = False
            else:
                next_stage = True

            if next_stage:
                # update stats for the prev iteration
                self.stats.update_average_configs_per_intensify(n_configs=self._chall_indx)

                # reset stats for the new iteration
                self._ta_time = 0
                self._chall_indx = 0
                self.num_run = 0

                self.iteration_done = True
                self.sh_iters += 1
                self.stage = 0
                self.run_tracker = {}
                self.configs_to_run = []
                self.fail_chal_offset = 0

                # randomize instance-seed pairs per successive halving run, if user specifies
                if self.instance_order == 'shuffle':
                    self.rs.shuffle(self.inst_seed_pairs)

        # to track configurations for the next stage
        self.success_challengers = set()  # successful configs
        self.do_not_advance_challengers = set()  # successful, but should not be advanced to the next budget/stage
        self.fail_challengers = set()  # capped/failed configs
        self.curr_inst_idx = {}
        self.running_challenger = None

    def _compare_configs(self,
                         incumbent: Configuration,
                         challenger: Configuration,
                         run_history: RunHistory,
                         log_traj: bool = True) -> typing.Optional[Configuration]:
        """
        Compares the challenger with current incumbent and returns the best configuration,
        based on the given incumbent selection design.

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : Configuration
            best configuration so far
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_traj : bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        typing.Optional[Configuration]
            incumbent configuration
        """

        if self.instance_as_budget:
            new_incumbent = super()._compare_configs(incumbent, challenger, run_history, log_traj)
            # if compare config returned none, then it is undecided. So return old incumbent
            new_incumbent = incumbent if new_incumbent is None else new_incumbent
            return new_incumbent

        # For real-valued budgets, compare configs based on the incumbent selection design
        curr_budget = self.all_budgets[self.stage]

        # incumbent selection: best on any budget
        if self.incumbent_selection == 'any_budget':
            new_incumbent = self._compare_configs_across_budgets(challenger=challenger,
                                                                 incumbent=incumbent,
                                                                 run_history=run_history,
                                                                 log_traj=log_traj)
            return new_incumbent

        # get runs for both configurations
        inc_runs = run_history.get_runs_for_config(incumbent, only_max_observed_budget=True)
        chall_runs = run_history.get_runs_for_config(challenger, only_max_observed_budget=True)

        if len(inc_runs) > 1:
            raise ValueError('Number of incumbent runs on budget %f must not exceed 1, but is %d',
                             inc_runs[0].budget, len(inc_runs))
        if len(chall_runs) > 1:
            raise ValueError('Number of challenger runs on budget %f must not exceed 1, but is %d',
                             chall_runs[0].budget, len(chall_runs))
        inc_run = inc_runs[0]
        chall_run = chall_runs[0]

        # incumbent selection: highest budget only
        if self.incumbent_selection == 'highest_budget':
            if chall_run.budget < self.max_budget:
                self.logger.debug('Challenger (budget=%.4f) has not been evaluated on the highest budget %.4f yet.',
                                  chall_run.budget, self.max_budget)
                return incumbent

        # incumbent selection: highest budget run so far
        if inc_run.budget > chall_run.budget:
            self.logger.debug('Incumbent evaluated on higher budget than challenger (%.4f > %.4f), '
                              'not changing the incumbent',
                              inc_run.budget, chall_run.budget)
            return incumbent
        if inc_run.budget < chall_run.budget:
            self.logger.debug('Challenger evaluated on higher budget than incumbent (%.4f > %.4f), '
                              'changing the incumbent',
                              chall_run.budget, inc_run.budget)
            if log_traj:
                # adding incumbent entry
                self.stats.inc_changed += 1
                new_inc_cost = run_history.get_cost(challenger)
                self.traj_logger.add_entry(train_perf=new_inc_cost, incumbent_id=self.stats.inc_changed,
                                           incumbent=challenger, budget=curr_budget)
            return challenger

        # incumbent and challenger were both evaluated on the same budget, compare them based on their cost
        chall_cost = run_history.get_cost(challenger)
        inc_cost = run_history.get_cost(incumbent)
        if chall_cost < inc_cost:
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) on budget %.4f.",
                             chall_cost, inc_cost, chall_run.budget)
            self._log_incumbent_changes(incumbent, challenger)
            new_incumbent = challenger
            if log_traj:
                # adding incumbent entry
                self.stats.inc_changed += 1  # first incumbent
                self.traj_logger.add_entry(train_perf=chall_cost, incumbent_id=self.stats.inc_changed,
                                           incumbent=new_incumbent, budget=curr_budget)
        else:
            self.logger.debug("Incumbent (%.4f) is at least as good as the challenger (%.4f) on budget %.4f.",
                              inc_cost, chall_cost, inc_run.budget)
            if log_traj and self.stats.inc_changed == 0:
                # adding incumbent entry
                self.stats.inc_changed += 1  # first incumbent
                self.traj_logger.add_entry(train_perf=inc_cost, incumbent_id=self.stats.inc_changed,
                                           incumbent=incumbent, budget=curr_budget)
            new_incumbent = incumbent

        return new_incumbent

    def _compare_configs_across_budgets(self,
                                        challenger: Configuration,
                                        incumbent: Configuration,
                                        run_history: RunHistory,
                                        log_traj: bool = True) -> typing.Optional[Configuration]:
        """
        compares challenger with current incumbent on any budget

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : Configuration
            best configuration so far
        run_history : smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        log_traj : bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        typing.Optional[Configuration]
            incumbent configuration
        """
        curr_budget = self.all_budgets[self.stage]

        # compare challenger and incumbent based on cost
        chall_cost = run_history.get_min_cost(challenger)
        inc_cost = run_history.get_min_cost(incumbent)
        if np.isfinite(chall_cost) and np.isfinite(inc_cost):
            if chall_cost < inc_cost:
                self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) for any budget.",
                                 chall_cost, inc_cost)
                self._log_incumbent_changes(incumbent, challenger)
                new_incumbent = challenger
                if log_traj:
                    # adding incumbent entry
                    self.stats.inc_changed += 1  # first incumbent
                    self.traj_logger.add_entry(train_perf=chall_cost, incumbent_id=self.stats.inc_changed,
                                               incumbent=new_incumbent, budget=curr_budget)
            else:
                self.logger.debug("Incumbent (%.4f) is at least as good as the challenger (%.4f) for any budget.",
                                  inc_cost, chall_cost)
                if log_traj and self.stats.inc_changed == 0:
                    # adding incumbent entry
                    self.stats.inc_changed += 1  # first incumbent
                    self.traj_logger.add_entry(train_perf=inc_cost, incumbent_id=self.stats.inc_changed,
                                               incumbent=incumbent, budget=curr_budget)
                new_incumbent = incumbent
        else:
            self.logger.debug('Non-finite costs from run history!')
            new_incumbent = incumbent

        return new_incumbent

    def _top_k(self,
               configs: typing.List[Configuration],
               run_history: RunHistory,
               k: int) -> typing.List[Configuration]:
        """
        Selects the top 'k' configurations from the given list based on their performance.

        This retrieves the performance for each configuration from the runhistory and checks
        that the highest budget they've been evaluated on is the same for each of the configurations.

        Parameters
        ----------
        configs: typing.List[Configuration]
            list of configurations to filter from
        run_history: smac.runhistory.runhistory.RunHistory
            stores all runs we ran so far
        k: int
            number of configurations to select

        Returns
        -------
        typing.List[Configuration]
            top challenger configurations, sorted in increasing costs
        """
        # extracting costs for each given configuration
        config_costs = {}
        # sample list instance-seed-budget key to act as base
        run_key = run_history.get_runs_for_config(configs[0], only_max_observed_budget=True)
        for c in configs:
            # ensuring that all configurations being compared are run on the same set of instance, seed & budget
            cur_run_key = run_history.get_runs_for_config(c, only_max_observed_budget=True)

            # Move to compare set -- get_runs_for_config queries form a dictionary
            # which is not an ordered structure. Some queries to that dictionary returned unordered
            # list which wrongly trigger the below if
            if set(cur_run_key) != set(run_key):
                raise ValueError(
                    'Cannot compare configs that were run on different instances-seeds-budgets: %s vs %s'
                    % (run_key, cur_run_key)
                )
            config_costs[c] = run_history.get_cost(c)

        configs_sorted = [k for k, v in sorted(config_costs.items(), key=lambda item: item[1])]
        # select top configurations only
        top_configs = configs_sorted[:k]
        return top_configs

    def _all_config_inst_seed_pairs_launched(
        self, run_history: RunHistory,
        activate_configuration_being_intensified: typing.Optional[Configuration],
    ) -> bool:
        """
        When running SH, M configs might require N instances. Before moving to the
        next stage, we need to make sure that tasks (each of the MxN jobs) are launched.

        This function returns a true if any M configs are pending or if N instance/seed are
        still remaining.

        Parameters
        ----------
        run_history : RunHistory
            stores all runs we ran so far
        activate_configuration_being_intensified: Optional[Configuration]
            The last configuration being actively processes by this intensifier

        Returns
        -------
            bool: whether a instance/pair of any of the M configurations for the current
                stage are pending
        """

        # 1: First we count the number of configurations that have been launched
        # We only submit a new configuration M if all instance-seed pairs of (M - 1)
        # have been proposed
        configurations_by_this_intensifier = [c for c, i, s, b in self.run_tracker]
        running_configs = set()
        for k, v in run_history.data.items():
            if run_history.ids_config[k.config_id] in configurations_by_this_intensifier:
                # We get all configurations launched by the current intensifier
                # regardless if status is RUNNING or not, to make it more robust
                running_configs.add(run_history.ids_config[k.config_id])

        # The total number of runs for this stage account for finished configurations
        # (success + failed + do not advance) + the offset + running but not finished
        # configurations. Also we account for the instances not launched for the
        # currently running configuration
        total_pending_configurations = max(0, self.n_configs_in_stage[self.stage] - (
            len(set().union(
                self.success_challengers,
                self.fail_challengers,
                self.do_not_advance_challengers,
                running_configs
            )) + self.fail_chal_offset))

        # 2: Second we have to account for the number of pending instances for the active
        # configuration. We assume for all (M - 1) configurations, all N instances-seeds
        # have been already been launched
        curr_budget = self.all_budgets[self.stage]
        if self.instance_as_budget:
            prev_budget = int(self.all_budgets[self.stage - 1]) if self.stage > 0 else 0
            curr_insts = self.inst_seed_pairs[int(prev_budget):int(curr_budget)]
        else:
            curr_insts = self.inst_seed_pairs

        if activate_configuration_being_intensified is None:
            # When a new stage begins, there is no active configuration.
            # Therefore activate_configuration_being_intensified is empty and all instances are
            # remaining
            pending_instances_to_launch = len(curr_insts)
        else:
            # self.curr_inst_idx - 1 is the last proposed instance/seed pair from get_next_run
            # But it is zero indexed, so (self.curr_inst_idx - 1) + 1 is the number of
            # configurations that we have proposed to run in total for the running
            # configuration via get_next_run
            pending_instances_to_launch = max(
                len(curr_insts) - self.curr_inst_idx[activate_configuration_being_intensified], 0)

        # If the there are any pending configuration, or instances/seed pending for the
        # active runner, we return a boolean
        return (total_pending_configurations + pending_instances_to_launch) <= 0


class SuccessiveHalving(ParallelScheduler):
    """ Races multiple challengers against an incumbent using Successive Halving method

    Implementation following the description in
    "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    Supplementary reference: http://proceedings.mlr.press/v80/falkner18a/falkner18a-supp.pdf

    Successive Halving intensifier (and Hyperband) can operate on two kinds of budgets:
    1. **'Instances' as budget**:
        When multiple instances are provided or when run objective is "runtime", this is the criterion used as budget
        for successive halving iterations i.e., the budget determines how many instances the challengers are evaluated
        on at a time. Top challengers for the next iteration are selected based on the combined performance across
        all instances used.

        If ``initial_budget`` and ``max_budget`` are not provided, then they are set to 1 and total number
        of available instances respectively by default.

    2. **'Real-valued' budget**:
        This is used when there is only one instance provided and when run objective is "quality",
        i.e., budget is a positive, real-valued number that can be passed to the target algorithm as an argument.
        It can be used to control anything by the target algorithm, Eg: number of epochs for training a neural network.

        ``initial_budget`` and ``max_budget`` are required parameters for this type of budget.

    Examples for successive halving (and hyperband) can be found here:
    * Runtime objective and multiple instances *(instances as budget)*: `examples/spear_qcp/SMAC4AC_SH_spear_qcp.py`
    * Quality objective and multiple instances *(instances as budget)*: `examples/SMAC4MF_sgd_instances.py`
    * Quality objective and single instance *(real-valued budget)*: `examples/SMAC4MF_mlp.py`

    This class instantiates `_SuccessiveHalving` objects on a need basis, that is, to
    prevent workers from being idle. The actual logic that implements the Successive halving method
    lies on the _SuccessiveHalving class.

    Parameters
    ----------
    stats: smac.stats.stats.Stats
        stats object
    traj_logger: smac.utils.io.traj_logging.TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : typing.Optional[int]
        cutoff of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    initial_budget : typing.Optional[float]
        minimum budget allowed for 1 run of successive halving
    max_budget : typing.Optional[float]
        maximum budget allowed for 1 run of successive halving
    eta : float
        'halving' factor after each iteration in a successive halving run. Defaults to 3
    num_initial_challengers : typing.Optional[int]
        number of challengers to consider for the initial budget. If None, calculated internally
    run_obj_time : bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    n_seeds : typing.Optional[int]
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : typing.Optional[str]
        how to order instances. Can be set to: [None, shuffle_once, shuffle]
        * None - use as is given by the user
        * shuffle_once - shuffle once and use across all SH run (default)
        * shuffle - shuffle before every SH run
    adaptive_capping_slackfactor : float
        slack factor of adpative capping (factor * adaptive cutoff)
    inst_seed_pairs : typing.List[typing.Tuple[str, int]], optional
        Do not set this argument, it will only be used by hyperband!
    min_chall: int
        minimal number of challengers to be considered (even if time_bound is exhausted earlier). This class will
        raise an exception if a value larger than 1 is passed.
    incumbent_selection: str
        How to select incumbent in successive halving. Only active for real-valued budgets.
        Can be set to: [highest_executed_budget, highest_budget, any_budget]
        * highest_executed_budget - incumbent is the best in the highest budget run so far (default)
        * highest_budget - incumbent is selected only based on the highest budget
        * any_budget - incumbent is the best on any budget i.e., best performance regardless of budget
    """

    def __init__(self,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[float] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 num_initial_challengers: typing.Optional[int] = None,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order: typing.Optional[str] = 'shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 inst_seed_pairs: typing.Optional[typing.List[typing.Tuple[str, int]]] = None,
                 min_chall: int = 1,
                 incumbent_selection: str = 'highest_executed_budget',
                 ) -> None:

        super().__init__(stats=stats,
                         traj_logger=traj_logger,
                         rng=rng,
                         instances=instances,
                         instance_specifics=instance_specifics,
                         cutoff=cutoff,
                         deterministic=deterministic,
                         run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor,
                         min_chall=min_chall)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # Successive Halving Hyperparameters
        self.n_seeds = n_seeds
        self.instance_order = instance_order
        self.inst_seed_pairs = inst_seed_pairs
        self.incumbent_selection = incumbent_selection
        self._instances = instances
        self._instance_specifics = instance_specifics
        self.initial_budget = initial_budget
        self.max_budget = max_budget
        self.eta = eta
        self.num_initial_challengers = num_initial_challengers

    def _get_intensifier_ranking(self, intensifier: AbstractRacer
                                 ) -> typing.Tuple[int, int]:
        """
        Given a intensifier, returns how advance it is.
        This metric will be used to determine what priority to
        assign to the intensifier

        Parameters
        ----------
        intensifier: AbstractRacer
            Intensifier to rank based on run progress

        Returns
        -------
        ranking: int
            the higher this number, the faster the intensifier will get
            the running resources. For hyperband we can use the
            sh_intensifier stage, for example
        tie_breaker: int
            The configurations that have been launched to break ties. For
            example, in the case of Successive Halving it can be the number
            of configurations launched
        """
        # For mypy -- we expect to work with Hyperband instances
        assert isinstance(intensifier, _SuccessiveHalving)

        # Each row of this matrix is id, stage, configs+instances for stage
        # We use sh.run_tracker as a cheap way to know how advanced the run is
        # in case of stage ties among successive halvers. sh.run_tracker is
        # also emptied each iteration
        stage = 0
        if hasattr(intensifier, 'stage'):
            # Newly created _SuccessiveHalving objects have no stage
            stage = intensifier.stage
        return stage, len(intensifier.run_tracker)

    def _add_new_instance(self, num_workers: int) -> bool:
        """
        Decides if it is possible to add a new intensifier instance,
        and adds it.
        If a new intensifier instance is added, True is returned, else False.

        Parameters
        ----------
        num_workers: int
            the maximum number of workers available
            at a given time.

        Returns
        -------
            Whether or not a successive halving instance was added
        """
        if len(self.intensifier_instances) >= num_workers:
            return False

        self.intensifier_instances[len(self.intensifier_instances)] = _SuccessiveHalving(
            stats=self.stats,
            traj_logger=self.traj_logger,
            rng=self.rs,
            instances=self._instances,
            instance_specifics=self._instance_specifics,
            cutoff=self.cutoff,
            deterministic=self.deterministic,
            initial_budget=self.initial_budget,
            max_budget=self.max_budget,
            eta=self.eta,
            num_initial_challengers=self.num_initial_challengers,
            run_obj_time=self.run_obj_time,
            n_seeds=self.n_seeds,
            instance_order=self.instance_order,
            adaptive_capping_slackfactor=self.adaptive_capping_slackfactor,
            inst_seed_pairs=self.inst_seed_pairs,
            min_chall=self.min_chall,
            incumbent_selection=self.incumbent_selection,
            identifier=len(self.intensifier_instances),
        )

        return True
