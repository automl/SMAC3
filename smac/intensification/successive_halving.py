import time
import logging
import typing
from collections import OrderedDict
from itertools import islice

import numpy as np

from smac.intensification.intensification import Intensifier
from smac.optimizer.objective import sum_cost
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import BudgetExhaustedException, CappedRunException, ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SuccessiveHalving(Intensifier):
    """Races multiple challengers against an incumbent using Successive Halving method

    Implementation from "BOHB: Robust and Efficient Hyperparameter Optimization at Scale" (Falkner et al. 2018)
    (refer supplementary)

    Parameters
    ----------
    tae_runner : smac.tae.execute_ta_run.ExecuteTARun Object
        target algorithm run executor
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
        slack factor of adpative capping (factor * adpative cutoff)
    """

    def __init__(self, tae_runner: ExecuteTARun,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: typing.Optional[int] = None,
                 deterministic: bool = False,
                 initial_budget: typing.Optional[float] = None,
                 max_budget: typing.Optional[float] = None,
                 eta: float = 3,
                 num_initial_challengers: typing.Optional[int] = None,
                 run_obj_time: bool = True,
                 n_seeds: typing.Optional[int] = None,
                 instance_order: typing.Optional[str] = 'shuffle_once',
                 adaptive_capping_slackfactor: float = 1.2,
                 **kwargs):

        super().__init__(tae_runner, stats, traj_logger, rng, instances,
                         instance_specifics=instance_specifics, cutoff=cutoff,
                         deterministic=deterministic, run_obj_time=run_obj_time,
                         adaptive_capping_slackfactor=adaptive_capping_slackfactor)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        # INSTANCES

        self.n_seeds = n_seeds if n_seeds else 1
        self.instance_order = instance_order

        # if instances are coming from Hyperband, skip the instance preprocessing section
        # it is already taken care by Hyperband
        if instances is not None and isinstance(instances[0], tuple):
            self.instances = list(instances)
        else:
            instances = [] if instances is None else instances
            # removing duplicates in the user provided instances
            instances = list(OrderedDict.fromkeys(instances))

            # NOTE Remove after solving how to handle multiple seeds and 1 instance
            if len(instances) == 1 and self.n_seeds > 1:
                raise NotImplementedError('This case (multiple seeds and 1 instance) cannot be handled yet!')

            # determine instance order
            if self.instance_order == 'shuffle_once':
                # randomize once
                self.rs.shuffle(instances)

            # set seed(s) for all SH runs
            # - currently user gives the number of seeds to consider
            if self.deterministic:
                seeds = [0]
            else:
                seeds = self.rs.randint(low=0, high=MAXINT, size=self.n_seeds)
            # storing instances & seeds as tuples
            self.instances = [(i, s) for s in seeds for i in instances]

        # successive halving parameters
        self._init_sh_params(initial_budget, max_budget, eta, num_initial_challengers)

        # adaptive capping
        if not self.runtime_as_budget and self.instance_order != 'shuffle' and self.run_obj_time:
            self.adaptive_capping = True
        else:
            self.adaptive_capping = False

    def _init_sh_params(self, initial_budget: typing.Optional[float],
                        max_budget: typing.Optional[float],
                        eta: float,
                        num_initial_challengers: typing.Optional[int]) -> None:
        """
        initialize Successive Halving parameters

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
        if not self.run_obj_time and len(self.instances) <= 1:
            # budget with cutoff
            if initial_budget is None or \
                    (max_budget is None and self.cutoff is None):
                raise ValueError("Successive Halving with runtime-cutoff as budget (i.e., only 1 instance) "
                                 "requires parameters initial_budget and max_budget/cutoff for intensification!")

            if self.cutoff is not None and max_budget is not None:
                self.logger.warning('Successive Halving with runtime-cutoff as budget: '
                                    'Both max budget (%d) and runtime-cutoff (%d) were provided. '
                                    'Max budget will be used.' % (max_budget, self.cutoff))

            self.initial_budget = initial_budget
            self.max_budget = max_budget if max_budget else self.cutoff
            self.runtime_as_budget = True

        else:
            # budget with instances
            if self.run_obj_time and len(self.instances) <= 1:
                self.logger.warning("Successive Halving has objective 'runtime' but only 1 instance-seed pair.")
            self.initial_budget = 1 if initial_budget is None else int(initial_budget)
            self.max_budget = len(self.instances) if max_budget is None else int(max_budget)
            self.runtime_as_budget = False

            if self.max_budget > len(self.instances):
                raise ValueError('Max budget cannot be greater than the number of instance-seed pairs')
            if self.max_budget < len(self.instances):
                self.logger.warning('Max budget (%d) does not include all instance-seed pairs (%d)' %
                                    (self.max_budget, len(self.instances)))

        budget_type = 'CUTOFF' if self.runtime_as_budget else 'INSTANCES'
        self.logger.info("Running Successive Halving with '%s' as budget. "
                         "Initial budget: %.2f, Max. budget: %.2f, eta: %.2f" %
                         (budget_type, self.initial_budget, self.max_budget, self.eta))

        # precomputing stuff for SH
        # max. no. of SH iterations possible given the budgets
        max_sh_iter = np.floor(np.log(self.max_budget / self.initial_budget) / np.log(self.eta))
        # initial number of challengers to sample
        if num_initial_challengers is None:
            self.num_initial_challengers = np.floor(self.eta ** max_sh_iter)
        else:
            self.num_initial_challengers = int(num_initial_challengers)
        # list of budgets that will be used in intensification
        self.all_budgets = self.max_budget * np.power(self.eta, -np.linspace(max_sh_iter, 0, max_sh_iter + 1))

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: typing.Optional[Configuration],
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: float = float(MAXINT),
                  log_traj: bool = True) -> typing.Tuple[Configuration, float]:
        """
        Running intensification via successive halving to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Implementation of successive halving (Jamieson & Talwalkar, 2016)

        Parameters
        ----------
        challengers : typing.List[Configuration]
            promising configurations
        incumbent : Configuration
            best configuration so far
        run_history : RunHistory
            stores all runs we ran so far
        aggregate_func: typing.Callable
            aggregate error across instances
        time_bound : float, optional (default=2 ** 31 - 1)
            time in [sec] available to perform intensify
        log_traj: bool
            whether to log changes of incumbents in trajectory

        Returns
        -------
        typing.Tuple[Configuration, float]
        """
        self.start_time = time.time()
        self._ta_time = 0
        self._chall_indx = 0
        self._num_run = 0
        first_run = True

        # challengers can be repeated only if optimizing across multiple seeds or changing instance orders every run
        if not (self.n_seeds > 1 or self.instance_order == 'shuffle'):
            used_configs = set(run_history.get_all_configs())
            chal_generator = (c for c in challengers if c not in used_configs)
        else:
            # converting to generator for consistency
            chal_generator = (c for c in challengers)
        # select first 'n' challengers
        curr_challengers = list(islice(chal_generator, int(self.num_initial_challengers)))

        # randomize instances per successive halving run, if user specifies
        all_instances = self.instances
        if self.instance_order == 'shuffle':
            self.rs.shuffle(all_instances)

        # calculating the incumbent's performance for adaptive capping
        #   - this check is required because there is no incumbent performance
        #     for the first ever 'intensify' run (from initial design)
        if incumbent is not None:
            inc_runs = run_history.get_runs_for_config(incumbent)
            inc_sum_cost = sum_cost(config=incumbent, instance_seed_budget_keys=inc_runs, run_history=run_history)
        else:
            inc_sum_cost = np.inf

        self.logger.debug('---' * 40)
        self.logger.debug('Successive Halving run begins. Budgets: %s' % self.all_budgets)

        # run intensification till budget is max
        for i, curr_budget in enumerate(self.all_budgets):

            self.logger.info('Running with budget [%.2f / %d] with %d challengers' %
                             (curr_budget, self.max_budget, len(curr_challengers)))
            # selecting instance subset for this budget, depending on the kind of budget
            prev_budget = self.all_budgets[i - 1] if i > 0 else 0
            available_insts = all_instances[int(prev_budget):int(curr_budget)] if not self.runtime_as_budget \
                else all_instances

            # determine 'k' for the next iteration - at least 1
            next_n_chal = max(1, int(len(curr_challengers) / self.eta))

            try:
                # Race all challengers
                curr_challengers = self._run_challengers(challengers=curr_challengers,
                                                         incumbent=incumbent,
                                                         instances=available_insts,
                                                         run_history=run_history,
                                                         budget=self.all_budgets[i],
                                                         inc_sum_cost=inc_sum_cost,
                                                         first_run=first_run)

                # if all challengers were capped, then stop intensification
                if not curr_challengers:
                    self.logger.info("All configurations have been eliminated by capping!"
                                     "Interrupting optimization run and returning current incumbent")
                    inc_perf = run_history.get_cost(incumbent)
                    return incumbent, inc_perf

            except BudgetExhaustedException:
                # Returning the final incumbent selected so far because we ran out of optimization budget
                self.logger.debug("Budget exhausted; "
                                  "Interrupting optimization run and returning current incumbent")
                inc_perf = run_history.get_cost(incumbent)
                return incumbent, inc_perf

            # selecting the top 'k' challengers for the next iteration
            curr_challengers = self._top_k(curr_challengers, run_history, k=next_n_chal)

            first_run = False

        # select best challenger from the SH run
        best_challenger = curr_challengers[0]
        self.logger.debug("Best challenger from successive halving run - %s" % (str(best_challenger)))

        # compare challenger with current incumbent
        if incumbent is None:  # first intensify run from initial design
            new_incumbent = best_challenger
            inc_perf = run_history.get_cost(best_challenger)
            self.logger.info("First Incumbent found! Cost of incumbent is (%.4f)" % inc_perf)
            self.logger.info("incumbent configuration: %s" % str(best_challenger))
            if log_traj:
                # adding incumbent entry
                self.stats.inc_changed += 1  # first incumbent
                self.traj_logger.add_entry(train_perf=inc_perf,
                                           incumbent_id=self.stats.inc_changed,
                                           incumbent=new_incumbent)

        else:
            new_incumbent = self._compare_configs(incumbent, best_challenger,
                                                  run_history, aggregate_func, log_traj)
            # if compare config returned none, then it is undecided. So return old incumbent
            new_incumbent = incumbent if new_incumbent is None else new_incumbent
            # getting new incumbent cost
            inc_perf = run_history.get_cost(new_incumbent)

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return new_incumbent, inc_perf

    def _run_challengers(self, challengers: typing.List[Configuration],
                         incumbent: Configuration,
                         instances: typing.List[typing.Tuple[str, int]],
                         run_history: RunHistory,
                         budget: float,
                         inc_sum_cost: float,
                         first_run: bool) -> typing.List[Configuration]:
        """
        Runs all the challengers for the given instance-seed pairs

        Parameters
        ----------
        challengers: typing.List[Configuration]
            List of challenger configurations to race
        incumbent:
            Current incumbent configuration
        instances: typing.Tuple[str, int]
            List of instance-seed pairs to use for racing challengers
        run_history: RunHistory
            Stores all runs we ran so far
        budget: float
            Successive Halving budget
        inc_sum_cost: float
            total sum cost of the incumbent (used to determine adaptive capping cutoff)
        first_run: bool
            to logs configurations to stats (set to true only for new configurations)

        Returns
        -------
        typing.List[Configuration]
            All challengers that were successfully executed, without being capped
        """

        # to track capped configurations & remove them later
        capped_configs = []

        # for every challenger generated, execute target algorithm
        for challenger in challengers:

            # for every instance in the instance subset
            self.logger.debug(" Running challenger  -  %s" % str(challenger))
            for instance, seed in instances:

                # selecting cutoff if running adaptive capping
                cutoff = self._adapt_cutoff(challenger=challenger,
                                            incumbent=incumbent,
                                            run_history=run_history,
                                            inc_sum_cost=inc_sum_cost)
                if cutoff is not None and cutoff <= 0:
                    # ran out of time to validate challenger
                    self.logger.debug("Stop challenger itensification due to adaptive capping.")
                    break

                # setting cutoff based on the type of budget & adaptive capping
                tae_cutoff = budget if self.runtime_as_budget else cutoff

                self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

                # run target algorithm for each instance-seed pair
                self.logger.debug("Execute target algorithm")
                try:
                    status, cost, dur, res = self.tae_runner.start(
                        config=challenger,
                        instance=instance,
                        seed=seed,
                        cutoff=tae_cutoff,
                        instance_specific=self.instance_specifics.get(instance, "0"),
                        capped=(self.cutoff is not None) and
                               (cutoff < self.cutoff)
                    )
                    self._ta_time += dur
                    self._num_run += 1

                except CappedRunException:
                    # We move on to the next configuration if we reach maximum cutoff i.e., capped
                    self.logger.debug("Budget exhausted by adaptive capping; "
                                      "Interrupting current challenger and moving on to the next one")
                    capped_configs.append(challenger)
                    break

            # count every challenger exactly once per SH run
            if first_run:
                self._chall_indx += 1

        # eliminate capped configuration from the race & reset capped_configs
        for c in capped_configs:
            challengers.remove(c)

        return challengers

    def _top_k(self, configs: typing.List[Configuration],
               run_history: RunHistory,
               k: int) -> typing.List[Configuration]:
        """
        selects the top 'k' configurations from the given list based on their performance in this budget

        Parameters
        ----------
        configs: typing.List[Configuration]
            list of configurations to filter from
        run_history: RunHistory
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
        run_key = run_history.get_runs_for_config(configs[0])
        for c in configs:
            # ensuring that all configurations being compared are run on the same set of instance, seed & budget
            cur_run_key = run_history.get_runs_for_config(c)
            if cur_run_key != run_key:
                raise AssertionError('Cannot compare configs that were run on different instances-seeds-budgets')
            config_costs[c] = run_history.get_cost(c)

        configs_sorted = sorted(config_costs, key=config_costs.get)
        # select top configurations only
        top_configs = configs_sorted[:k]
        return top_configs
