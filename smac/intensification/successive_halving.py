import sys
import time
import logging
import typing
from collections import OrderedDict

import numpy as np

from smac.intensification.intensification import Intensifier
from smac.optimizer.ei_optimization import ChallengerList
from smac.optimizer.objective import sum_cost
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT, MAX_CUTOFF
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory, InstSeedKey
from smac.tae.execute_ta_run import StatusType, BudgetExhaustedException, CappedRunException, ExecuteTARun
from smac.utils.io.traj_logging import TrajLogger

__author__ = "Ashwin Raaghav Narayanan"
__copyright__ = "Copyright 2018, ML4AAD"
__license__ = "3-clause BSD"


class SuccessiveHalving(Intensifier):
    """Races multiple challengers against an incumbent using Successive Halving method

    Parameters
    ----------
    tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
        target algorithm run executor
    stats: Stats
        stats object
    traj_logger: TrajLogger
        TrajLogger object to log all new incumbents
    rng : np.random.RandomState
    instances : typing.List[str]
        list of all instance ids
    instance_specifics : typing.Mapping[str,np.ndarray]
        mapping from instance name to instance specific string
    cutoff : int
        runtime cutoff of TA runs
    deterministic : bool
        whether the TA is deterministic or not
    min_budget : int
        minimum budget allowed for 1 run of successive halving
    max_budget : int
        maximum budget allowed for 1 run of successive halving
    eta : int
        'halving' factor after each iteration in a successive halving run
    run_obj_time : bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    n_seeds : int
        Number of seeds to use, if TA is not deterministic. Defaults to None, i.e., seed is set as 0
    instance_order : str
        how to order instances. Can be set to:
        None - use as is given by the user
        random - shuffle once and use across all SH run (default)
        budget_random - shuffle before every SH run
    adaptive_capping_slackfactor : float
        slack factor of adpative capping (factor * adpative cutoff)
    """

    def __init__(self, tae_runner: ExecuteTARun,
                 stats: Stats,
                 traj_logger: TrajLogger,
                 rng: np.random.RandomState,
                 instances: typing.List[str],
                 instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 cutoff: int = MAX_CUTOFF,
                 deterministic: bool = False,
                 min_budget: float = None,
                 max_budget: float = None,
                 eta: float = 2,
                 run_obj_time: bool = True,
                 n_seeds: int = None,
                 instance_order='random',
                 adaptive_capping_slackfactor: float = 1.2, **kwargs):

        super().__init__(tae_runner, stats, traj_logger, rng, instances)

        self.logger = logging.getLogger(
            self.__module__ + "." + self.__class__.__name__)

        self.stats = stats
        self.traj_logger = traj_logger

        # general attributes
        self.rs = rng
        self._num_run = 0
        self._chall_indx = 0
        self._ta_time = 0
        self._min_time = 10 ** -5
        self.start_time = None

        # scenario info
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae_runner = tae_runner

        # instance related info
        if instances is None:
            instances = []
        # removing duplicates in the user provided instances
        self.instances = list(OrderedDict.fromkeys(instances))
        if instance_specifics is None:
            self.instance_specifics = {}
        else:
            self.instance_specifics = instance_specifics

        # determine instance order
        self.instance_order = instance_order
        if self.instance_order == 'random':
            # randomize once
            np.random.shuffle(self.instances)

        # set seed(s) for all SH runs
        # - currently user gives the number of seeds to consider
        self.n_seeds = n_seeds
        if self.deterministic:
            seeds = [0]
        else:
            if self.n_seeds is None:
                seeds = [self.rs.randint(low=0, high=MAXINT)]
            else:
                seeds = self.rs.randint(low=0, high=MAXINT, size=self.n_seeds)
        # storing instances & seeds as tuples
        self.instances = [InstSeedKey(i, s) for s in seeds for i in self.instances]

        # successive halving parameters
        if eta <= 1:
            raise ValueError('eta must be greater than 1')
        self.cutoff = cutoff
        self.eta = eta

        # determine budgets from given instances if none mentioned
        # - if only 1 instance/no instances were provided, then use cutoff as budget
        # - else, use instances as budget
        if len(self.instances) <= 1:
            # budget with cutoff
            # cannot run successive halving with cutoff as budget if budget limits are not provided!
            if min_budget is None or \
                    (max_budget is None and cutoff is None):
                raise ValueError("Successive Halving with cutoff as budget (i.e., only 1 instance) requires parameters "
                                 "min_budget and max_budget/cutoff for intensification!")
            self.min_budget = min_budget
            self.max_budget = self.cutoff if max_budget is None else max_budget
            self.budget_type = 'cutoff'
        else:
            # budget with instances
            self.min_budget = 1 if min_budget is None else int(min_budget)
            self.max_budget = len(self.instances) if max_budget is None else int(max_budget)
            self.budget_type = 'instance'

            # max budget cannot be greater than number of instance-seed pairs
            if self.max_budget > len(self.instances):
                raise ValueError('Max budget cannot be greater than the number of instance-seed pairs')

        self.logger.info("Running Successive Halving with '%s' as budget" % self.budget_type.upper())

        # number configurations to consider for a full successive halving iteration
        self.max_sh_iter = np.floor(np.log(self.max_budget / self.min_budget) / np.log(eta))
        self.initial_challengers = int(np.round(self.eta ** (self.max_sh_iter + 1)))

        # for adaptive capping
        if self.budget_type == 'instance' and self.instance_order != 'budget_random' and self.run_obj_time:
            self.adaptive_capping = True
        else:
            self.adaptive_capping = False
        self.adaptive_capping_slackfactor = adaptive_capping_slackfactor

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: typing.Optional[Configuration],
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: float = float(MAXINT),
                  log_traj: bool = True):
        """
        Running intensification via successive halving to determine the incumbent configuration.
        *Side effect:* adds runs to run_history

        Implementation of successive halving by Jamieson & Talwalkar (2016)

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
        incumbent: Configuration()
            current (maybe new) incumbent configuration
        inc_perf: float
            empirical performance of incumbent configuration
        """
        self.start_time = time.time()
        self._ta_time = 0
        self._chall_indx = 0
        self._num_run = 0

        if time_bound < self._min_time:
            raise ValueError("time_bound must be >= %f" % self._min_time)

        if isinstance(challengers, ChallengerList):
            challengers = challengers.challengers

        # selecting first 'n' challengers for the current run, based on eta
        curr_challengers = challengers[:self.initial_challengers]
        # challengers can be repeated only if optimizing across multiple seeds
        if self.n_seeds is None or self.n_seeds == 1:
            curr_challengers = set(curr_challengers) - set(run_history.get_all_configs())

        # randomize instances per successive halving run, if user specifies
        all_instances = self.instances
        if self.instance_order == 'budget_random':
            np.random.shuffle(all_instances)

        # calculating the incumbent's performance for adaptive capping
        # this check is required because there is no initial run to get the incumbent performance
        if incumbent is not None:
            inc_runs = run_history.get_runs_for_config(incumbent)
            inc_sum_cost = sum_cost(config=incumbent, instance_seed_pairs=inc_runs, run_history=run_history)
        else:
            inc_sum_cost = np.inf

        # selecting the 1st budget for 1st round of successive halving
        sh_iter = self.max_sh_iter
        budget = self.max_budget / (self.eta ** sh_iter)
        budget = max(budget, self.min_budget)
        prev_budget = 0

        first_run = True

        self.logger.debug('---' * 40)
        self.logger.debug('Successive Halving run begins.')

        # run intesification till budget is max
        while budget <= self.max_budget:

            self.logger.info('Running with budget [%.2f / %d] with %d challengers' % (budget, self.max_budget,
                                                                                      len(curr_challengers)))
            # selecting instance subset for this budget, depending on the kind of budget
            if self.budget_type == 'instance':
                # run only on new instances
                available_insts = all_instances[int(prev_budget):int(budget)]
            else:
                available_insts = all_instances

            # determine 'k' for the next iteration
            next_n_chal = int(np.round(len(curr_challengers) / self.eta))

            try:
                # Race all challengers
                curr_challengers = self._race_challengers(challengers=curr_challengers,
                                                          incumbent=incumbent,
                                                          instances=available_insts,
                                                          run_history=run_history,
                                                          budget=budget,
                                                          inc_sum_cost=inc_sum_cost,
                                                          first_run=first_run)

                # if all challengers were capped, then stop intensification
                if not curr_challengers:
                    self.logger.info("All configurations have been eliminated!"
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
            if next_n_chal > 0:
                curr_challengers = self._top_k(curr_challengers, run_history, k=next_n_chal)
            else:
                # returning at least 1 configuration for next iteration since the incumbent has to run on all instances
                self.logger.info('Ran out of configurations! Returning the best from what was seen so far.')
                curr_challengers = self._top_k(curr_challengers, run_history, k=1)

            first_run = False

            # increase budget for next iteration
            prev_budget = budget
            budget = budget * self.eta

        # select best challenger from the SH run
        best_challenger = curr_challengers[0]
        self.logger.debug("Best challenger from successive halving run - %s" % (str(best_challenger)))

        # compare challenger with current incumbent
        if incumbent is None:
            new_incumbent = best_challenger
            inc_perf = run_history.get_cost(best_challenger)
            self.logger.info("First Incumbent found! Cost of incumbent is (%.4f)" % inc_perf)
            self.logger.info("incumbent configuration: %s" % str(best_challenger))
        else:
            new_incumbent = self._compare_configs(incumbent, best_challenger,
                                                  run_history, aggregate_func, log_traj)
            if new_incumbent is None:
                # if compare config returned none, then it is undecided. So returns incumbent
                new_incumbent = incumbent
            # getting new incumbent cost
            inc_perf = run_history.get_cost(new_incumbent)

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return new_incumbent, inc_perf

    def _race_challengers(self, challengers: typing.List[Configuration],
                          incumbent: Configuration,
                          instances: typing.List[InstSeedKey],
                          run_history: RunHistory,
                          budget: float,
                          inc_sum_cost: float,
                          first_run: bool):
        """
        Aggressively race challengers for the given instances

        Parameters
        ----------
        challengers: typing.List[Configuration]
            List of challenger configurations to race
        incumbent:
            Current incumbent configuration
        instances: typing.List[InstSeedKey]
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
        challengers: typing.List[Configuration]
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
                    # no time to validate challenger
                    self.logger.debug("Stop challenger itensification due to adaptive capping.")
                    break

                # setting cutoff based on the type of budget & adaptive capping
                tae_cutoff = min(cutoff, budget) if self.budget_type == 'cutoff' else cutoff

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
        _ = [challengers.remove(c) for c in capped_configs]

        return challengers

    def _top_k(self, configs: typing.List[Configuration],
               run_history: RunHistory,
               k: int):
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
        challengers: typing.List[Configuration]
            top challenger configurations, sorted in increasing costs
        """
        # NOTE run_history overwrites the cost for runs since it doesnt record each budget (cutoff) separately
        # extracting costs for each given configuration
        config_costs = {}
        for c in configs:
            config_costs[c] = run_history.get_cost(c)

        configs_sorted = sorted(config_costs, key=config_costs.get)
        # select top configurations only
        top_configs = configs_sorted[:k]
        return top_configs
