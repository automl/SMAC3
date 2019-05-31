import sys
import time
import logging
import typing

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
    deterministic: bool
        whether the TA is deterministic or not
    run_obj_time: bool
        whether the run objective is runtime or not (if true, apply adaptive capping)
    always_race_against: Configuration
        if incumbent changes race this configuration always against new incumbent;
        can sometimes prevent over-tuning
    use_ta_time_bound: bool,
        if true, trust time reported by the target algorithms instead of
        measuring the wallclock time for limiting the time of intensification
    run_limit : int
        Maximum number of target algorithm runs per call to intensify.
    maxR : int
        Maximum number of runs per config (summed over all calls to
        intensifiy).
    minR : int
        Minimum number of run per config (summed over all calls to
        intensify).
    adaptive_capping_slackfactor: float
        slack factor of adpative capping (factor * adpative cutoff)
    min_chall: int
        minimal number of challengers to be considered
        (even if time_bound is exhausted earlier)
    """

    def __init__(self, tae_runner: ExecuteTARun, stats: Stats, traj_logger: TrajLogger, rng: np.random.RandomState,
                 instances: typing.List[str], max_budget: int = None, min_budget: int = None, eta: float = 2,
                 cutoff: int = MAX_CUTOFF, instance_specifics: typing.Mapping[str, np.ndarray] = None,
                 deterministic: bool = False, run_obj_time: bool = True, n_seeds: int = None, instance_order='random',
                 adaptive_capping_slackfactor: float = 1.2,
                 always_race_against: Configuration = None, run_limit: int = MAXINT, use_ta_time_bound: bool = False,
                 minR: int = 1, maxR: int = 2000, min_chall: int = 2):

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
        self.instances = list(set(instances))
        if instance_specifics is None:
            self.instance_specifics = {}
        else:
            self.instance_specifics = instance_specifics

        # determine instance order
        self.instance_order = instance_order
        if instance_order == 'random':
            # randomize once
            np.random.shuffle(self.instances)
        # TODO: find better way to do this - or can it just be the order in which user uploads the data?
        # elif isinstance(self.instances, list):
        #     # use user given order
        #     assert len(self.instances) == len(instance_order), \
        #         'instance_order (%d) should be same length as instances (%d)' % (len(instance_order),
        #                                                                          len(self.instances))
        #     self.instances = self.instances[instance_order]

        # set seed(s) for all SH runs
        # TODO (multiple) seeds will be assigned based on budget - how many new seeds to select?
        # - currently user gives the number of seeds to consider
        self.n_seeds = n_seeds
        if self.deterministic:
            seeds = [0]
        else:
            if self.n_seeds is None:
                seeds = [self.rs.randint(low=0, high=MAXINT, size=1)[0]]
            else:
                seeds = [self.rs.randint(low=0, high=MAXINT, size=n_seeds)[0]]
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
        if len(instances) <= 1:
            # budget with cutoff
            self.min_budget = self._min_time if min_budget is None else min_budget
            self.max_budget = self.cutoff if max_budget is None else max_budget
            self.budget_type = 'cutoff'
        else:
            # budget with instances
            self.min_budget = 1 if min_budget is None else min_budget
            self.max_budget = len(self.instances) if max_budget is None else max_budget
            self.budget_type = 'instance'

        # number configurations to consider for a full successive halving iteration
        self.max_sh_iter = np.floor(np.log(self.max_budget / self.min_budget) / np.log(eta))
        self.initial_challengers = int(np.round(self.eta ** (self.max_sh_iter + 1)))

        # for adaptive capping
        if self.budget_type == 'instance' and self.instance_order != 'budget_random' and self.run_obj_time:
            self.adaptive_capping = True
        else:
            self.adaptive_capping = False
        self.adaptive_capping_slackfactor = adaptive_capping_slackfactor

        # self.always_race_against = always_race_against
        # self.run_limit = run_limit
        # self.maxR = maxR
        # self.minR = minR
        # if self.run_limit < 1:
        #     raise ValueError("run_limit must be > 1")
        # self.min_chall = min_chall
        # self.use_ta_time_bound = use_ta_time_bound

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: Configuration,
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
        # in the 1st SH run, the incumbent is also added to the challenger configuration list to run,
        # since it was not run before, (set incumbent to None for a fair comparison)
        if len(run_history.get_runs_for_config(incumbent)) == 0:
            challengers = [incumbent] + challengers
            incumbent = None

        # selecting first 'n' challengers for the current run, based on eta
        curr_challengers = challengers[:self.initial_challengers]
        # challengers can repeated only if optimizing across multiple seeds
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
        budget = int(max(budget, self.min_budget))
        prev_budget = 0

        first_run = False

        self.logger.debug('---' * 40)
        self.logger.debug('Successive Halving run begins.')

        # run intesification till budget is max
        while budget <= self.max_budget:

            self.logger.info('Running with budget [%d / %d] with %d challengers' % (budget, self.max_budget,
                                                                                    len(curr_challengers)))
            # selecting instance subset for this budget, depending on the kind og budget
            if self.budget_type == 'instance':
                # run only on new instances
                available_insts = all_instances[prev_budget:budget]
            else:
                available_insts = all_instances

            # for every challenger generated, execute target algorithm
            for challenger in curr_challengers:

                # for every instance in the instance subset
                self.logger.debug(" Running challenger  -  %s" % str(challenger))
                for instance, seed in available_insts:

                    # selecting cutoff if running adaptive capping
                    cutoff = self._adapt_cutoff(challenger=challenger,
                                                incumbent=incumbent,
                                                run_history=run_history,
                                                inc_sum_cost=inc_sum_cost)
                    if cutoff is not None and cutoff <= 0:
                        # no time to validate challenger
                        self.logger.debug("Stop challenger itensification due to adaptive capping.")
                        break

                    self.logger.debug('Cutoff for challenger: %s' % str(cutoff))

                    # run target algorithm for each instance-seed pair
                    self.logger.debug("Execute target algorithm")
                    try:
                        if self.budget_type == 'cutoff':
                            status, cost, dur, res = self.tae_runner.start(
                                config=challenger,
                                instance=instance,
                                seed=seed,
                                cutoff=budget,
                                instance_specific=self.instance_specifics.get(instance, "0"))
                        else:
                            status, cost, dur, res = self.tae_runner.start(
                                config=challenger,
                                instance=instance,
                                seed=seed,
                                cutoff=cutoff,
                                instance_specific=self.instance_specifics.get(instance, "0"),
                                capped=(self.cutoff is not None) and
                                       (cutoff < self.cutoff)
                            )
                        self._ta_time += dur

                    except CappedRunException:
                        # We move on to the next configuration if we reach maximum cutoff i.e., capped
                        self.logger.debug("Budget exhausted by adaptive capping; "
                                          "Interrupting current challenger and moving on to the next one")
                        break

                    except BudgetExhaustedException:
                        # TODO revisit what to return if budget exhausted. What is the use of inc_perf in smbo?
                        # Returning the final incumbent selected so far because we ran out of optimization budget
                        self.logger.debug("Budget exhausted; "
                                          "Interrupting optimization run and returning current incumbent")
                        return incumbent, None

                    self._num_run += 1

                if not first_run:
                    first_run = True
                    self._chall_indx += 1

            # selecting the top 'k' challengers for the next iteration
            next_n_chal = int(np.round(len(curr_challengers) / self.eta))
            if next_n_chal >= 1:
                curr_challengers = self._top_k(curr_challengers, run_history, k=next_n_chal)
            else:
                # returning at least 1 configuration for next iteration since the incumbent has to run on all instances
                self.logger.debug('Ran out of configurations! Returning the best from what was seen so far.')
                curr_challengers = self._top_k(curr_challengers, run_history, k=1)

            # increase budget for next iteration
            prev_budget = budget
            budget = budget * self.eta

        # select best challenger from the SH run
        best_challenger = curr_challengers[0]
        self.logger.debug("Best challenger from successive halving run - %s" % (str(best_challenger)))

        # compare challenger with current incumbent
        new_incumbent, inc_perf = self._compare_configs(incumbent, best_challenger, run_history, log_traj)

        return new_incumbent, inc_perf

    def _top_k(self, configs: typing.List[Configuration], run_history: RunHistory, k: int):
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
        # extracting costs for each given configuration
        config_costs = {}
        for c in configs:
            runs = run_history.get_runs_for_config(c)
            perf = run_history.aggregate_func(c, run_history, runs)
            config_costs[c] = perf

        configs_sorted = sorted(config_costs, key=config_costs.get)
        # select top configurations only
        top_configs = configs_sorted[:k]
        return top_configs

    def _adapt_cutoff(self, challenger: Configuration,
                      incumbent: Configuration,
                      run_history: RunHistory,
                      inc_sum_cost: float):

        """Adaptive capping:
        Compute cutoff based on time so far used for incumbent
        and reduce cutoff for next run of challenger accordingly

        !Only applicable if self.adaptive capping i.e.,
            - objective is runtime
            - sucessive halving is on instances and not cutoff
            - instance order is not changed per budget

        Parameters
        ----------
        challenger : Configuration
            Configuration which challenges incumbent
        incumbent : Configuration
            Best configuration so far
        run_history : RunHistory
            Stores all runs we ran so far
        inc_sum_cost: float
            Sum of runtimes of all incumbent runs

        Returns
        -------
        cutoff: int
            Adapted cutoff
        """

        if not self.adaptive_capping:
            return self.cutoff

        # cost used by challenger for going over all its runs
        # should be subset of runs of incumbent (not checked for efficiency
        # reasons)
        chall_inst_seeds = run_history.get_runs_for_config(challenger)
        chal_sum_cost = sum_cost(config=challenger,
                                 instance_seed_pairs=chall_inst_seeds,
                                 run_history=run_history)
        cutoff = min(self.cutoff,
                     inc_sum_cost * self.adaptive_capping_slackfactor -
                     chal_sum_cost
                     )
        return cutoff

    def _compare_configs(self, incumbent: Configuration, challenger: Configuration, run_history: RunHistory,
                         log_traj: bool, **kwargs):
        """
        compares challenger and incumbent based on all runs seen so far

        Parameters
        ----------
        challenger : Configuration
            promising configuration
        incumbent : Configuration
            best configuration so far
        run_history: RunHistory
            stores all runs we ran so far
        log_traj : bool
            flag to specify if incumbent trajectory is to be recorded
        Returns
        -------
        new_incumbent: Configuration
            best configuration among challenger and incumbent
            :param **kwargs:
        """
        # NOTE run_history overwrites the cost for runs since it doesnt record each budget (cutoff) separately (yet?)
        # get incumbent & challenger performance
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = run_history.aggregate_func(incumbent, run_history, inc_runs)
        chal_runs = run_history.get_runs_for_config(challenger)
        chal_perf = run_history.aggregate_func(challenger, run_history, chal_runs)

        if incumbent == challenger:
            self.logger.info("Incumbent remains unchanged.")
            return incumbent, inc_perf
        elif not np.isnan(inc_perf) and inc_perf < chal_perf:
            self.logger.info("Incumbent remains unchanged. Incumbent is still better (%.4f) than challenger "
                             "(%.4f) in this run" % (inc_perf, chal_perf))
            return incumbent, inc_perf
        else:
            self.logger.info("Incumbent changed! Challenger (%.4f) is better than "
                             "incumbent (%.4f)" % (chal_perf, inc_perf))
            if log_traj:
                self.stats.inc_changed += 1
                self.traj_logger.add_entry(train_perf=chal_perf,
                                           incumbent_id=self.stats.inc_changed,
                                           incumbent=challenger)
            return challenger, chal_perf
