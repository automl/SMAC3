import sys
import time
import copy
import logging
import typing
from collections import Counter
from collections import OrderedDict

import numpy as np

from smac.smbo.objective import sum_cost
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT, MAX_CUTOFF
from smac.configspace import Configuration
from smac.runhistory.runhistory import RunHistory
from smac.tae.execute_ta_run import StatusType

__author__ = "Katharina Eggensperger, Marius Lindauer"
__copyright__ = "Copyright 2017, ML4AAD"
__license__ = "3-clause BSD"


class Intensifier(object):

    '''
        takes challenger and incumbents and performs intensify
    '''

    def __init__(self, tae_runner, stats, traj_logger, rng, instances,
                 instance_specifics=None, cutoff=MAX_CUTOFF, deterministic=False,
                 run_obj_time=True, run_limit=MAXINT, minR=1, maxR=2000):
        '''
        Constructor

        Parameters
        ----------
        tae_runner : tae.executre_ta_run_*.ExecuteTARun* Object
            target algorithm run executor
        stats: Stats()
            stats object            
        traj_logger: TrajLogger()
            TrajLogger object to log all new incumbents
        rng : np.random.RandomState
        instances : list
            list of all instance ids
        instance_specifics : dict
            mapping from instance name to instance specific string
        cutoff : int
            runtime cutoff of TA runs
        deterministic: bool
            whether the TA is deterministic or not
        run_obj_time: bool
            whether the run objective is runtime or not (if true, apply adaptive capping)
        run_limit : int
            maximum number of runs
        maxR : int
            maximum number of runs per config
        minR : int
            minimum number of run per config
        '''
        self.stats = stats
        self.traj_logger = traj_logger
        # general attributes
        if instances is None:
            instances = []
        self.instances = set(instances)
        if instance_specifics is None:
            self.instance_specifics = {}
        else:
            self.instance_specifics = instance_specifics
        self.logger = logging.getLogger("intensifier")
        self.run_limit = run_limit
        self.maxR = maxR
        self.minR = minR
        self.rs = rng

        # scenario info
        self.cutoff = cutoff
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae_runner = tae_runner

        self.Adaptive_Capping_Slackfactor = 1.2

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

        self._num_run = 0
        self._chall_indx = 0

    def intensify(self, challengers: typing.List[Configuration],
                  incumbent: Configuration,
                  run_history: RunHistory,
                  aggregate_func: typing.Callable,
                  time_bound: int=MAXINT):
        '''
            running intensification to determine the incumbent configuration
            Side effect: adds runs to run_history

            Implementation of Procedure 2 in Hutter et al. (2011).

            Parameters
            ----------

            challengers : typing.List[Configuration]
                promising configurations
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate performance across instances
            time_bound : int, optional (default=2 ** 31 - 1)
                time in [sec] available to perform intensify

            Returns
            -------
            incumbent: Configuration()
                current (maybe new) incumbent configuration
            inc_perf: float
                empirical performance of incumbent configuration 
        '''

        self.start_time = time.time()

        if time_bound < 0.01:
            raise ValueError("time_bound must be >= 0.01")

        self._num_run = 0
        self._chall_indx = 0

        # Line 1 + 2
        for challenger in challengers:
            if challenger == incumbent:
                self.logger.warning(
                    "Challenger was the same as the current incumbent; Skipping challenger")
                continue

            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug(
                    "Configuration origin: %s", challenger.origin)

            # Lines 3-7
            self._add_inc_run(incumbent=incumbent, run_history=run_history)

            # Lines 8-17
            incumbent = self._race_challenger(challenger=challenger,
                                              incumbent=incumbent,
                                              run_history=run_history,
                                              aggregate_func=aggregate_func)

            if self._chall_indx > 1 and self._num_run > self.run_limit:
                self.logger.debug(
                    "Maximum #runs for intensification reached")
                break
            elif self._chall_indx > 1 and time.time() - self.start_time - time_bound >= 0:
                self.logger.debug("Timelimit for intensification reached ("
                                  "used: %f sec, available: %f sec)" % (
                                      time.time() - self.start_time, time_bound))
                break

        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = aggregate_func(incumbent, run_history, inc_runs)
        self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
            len(inc_runs), inc_perf))

        self.stats.update_average_configs_per_intensify(
            n_configs=self._chall_indx)

        return incumbent, inc_perf

    def _add_inc_run(self, incumbent: Configuration, run_history: RunHistory):
        '''
            add new run for incumbent
            Side effect: adds runs to <run_history>

            Parameters
            ----------
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
        '''

        inc_runs = run_history.get_runs_for_config(incumbent)

        # Line 3
        # First evaluate incumbent on a new instance
        if len(inc_runs) < self.maxR:
            while True:
                # Line 4
                # find all instances that have the most runs on the inc
                inc_runs = run_history.get_runs_for_config(incumbent)
                inc_inst = [s.instance for s in inc_runs]
                inc_inst = list(Counter(inc_inst).items())
                inc_inst.sort(key=lambda x: x[1], reverse=True)
                try:
                    max_runs = inc_inst[0][1]
                except IndexError:
                    self.logger.debug("No run for incumbent found")
                    max_runs = 0
                inc_inst = set([x[0] for x in inc_inst if x[1] == max_runs])

                available_insts = (self.instances - inc_inst)

                # if all instances were used n times, we can pick an instances
                # from the complete set again
                if not self.deterministic and not available_insts:
                    available_insts = self.instances

                # Line 6 (Line 5 is further down...)
                if self.deterministic:
                    next_seed = 0
                else:
                    next_seed = self.rs.randint(low=0, high=MAXINT,
                                                size=1)[0]

                if available_insts:
                    # Line 5 (here for easier code)
                    next_instance = self.rs.choice(list(available_insts))
                    # Line 7
                    self.logger.debug("Add run of incumbent")
                    status, cost, dur, res = self.tae_runner.start(
                        config=incumbent,
                        instance=next_instance,
                        seed=next_seed,
                        cutoff=self.cutoff,
                        instance_specific=self.instance_specifics.get(next_instance, "0"))

                    self._num_run += 1
                else:
                    self.logger.debug(
                        "No further instance-seed pairs for incumbent available.")
                    break

                inc_runs = run_history.get_runs_for_config(incumbent)
                # Termination condition; after exactly one run, this checks
                # whether further runs are necessary due to minR
                if len(inc_runs) >= self.minR or len(inc_runs) >= self.maxR:
                    break

    def _race_challenger(self, challenger: Configuration, incumbent: Configuration, run_history: RunHistory,
                         aggregate_func: typing.Callable):
        '''
            aggressively race challenger against incumbent

            Parameters
            ----------
            challenger : Configuration
                configuration which challenges incumbent
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate performance across instances

            Returns
            -------
            new_incumbent: Configuration
                either challenger or incumbent
        '''
        # at least one run of challenger
        # to increase chall_indx counter
        first_run = False

        # Line 8
        N = max(1, self.minR)

        inc_inst_seeds = set(run_history.get_runs_for_config(incumbent))
        # Line 9
        while True:
            chall_inst_seeds = set(run_history.get_runs_for_config(challenger))

            # Line 10
            missing_runs = list(inc_inst_seeds - chall_inst_seeds)

            # Line 11
            self.rs.shuffle(missing_runs)
            to_run = missing_runs[:min(N, len(missing_runs))]
            # Line 13 (Line 12 comes below...)
            missing_runs = missing_runs[min(N, len(missing_runs)):]

            # for adaptive capping
            # because of efficieny computed here
            inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))
            # cost used by incumbent for going over all runs in inst_seed_pairs
            inc_sum_cost = sum_cost(config=incumbent, instance_seed_pairs=inst_seed_pairs,
                                    run_history=run_history)

            # Line 12
            # Run challenger on all <config,seed> to run
            for instance, seed in to_run:

                cutoff = self._adapt_cutoff(challenger=challenger, incumbent=incumbent,
                                            run_history=run_history, inc_sum_cost=inc_sum_cost)
                if cutoff is not None and cutoff <= 0:  # no time to validate challenger
                    self.logger.debug(
                        "Stop challenger itensification due to adaptive capping.")
                    # challenger performs worse than incumbent
                    return incumbent

                if not first_run:
                    first_run = True
                    self._chall_indx += 1

                self.logger.debug("Add run of challenger")
                status, cost, dur, res = self.tae_runner.start(
                    config=challenger,
                    instance=instance,
                    seed=seed,
                    cutoff=cutoff,
                    instance_specific=self.instance_specifics.get(instance, "0"))
                self._num_run += 1

                if status == StatusType.ABORT:
                    self.logger.debug(
                        "TAE signaled ABORT -- stop intensification on challenger")
                    return incumbent

            new_incumbent = self._compare_configs(
                incumbent=incumbent, challenger=challenger, run_history=run_history,
                aggregate_func=aggregate_func)
            if new_incumbent == incumbent:
                break
            elif new_incumbent == challenger:
                incumbent = challenger
                break
            else:  # Line 17
                # challenger is not worse, continue
                N = 2 * N

        return incumbent

    def _adapt_cutoff(self, challenger: Configuration,
                      incumbent: Configuration,
                      run_history: RunHistory,
                      inc_sum_cost: float):
        '''
            adaptive capping: 
            compute cutoff based on time so far used for incumbent
            and reduce cutoff for next run of challenger accordingly

            !Only applicable if self.run_obj_time

            !runs on incumbent should be superset of the runs performed for the challenger

            Parameters
            ----------
            challenger : Configuration
                configuration which challenges incumbent
            incumbent : Configuration
                best configuration so far
            run_history : RunHistory
                stores all runs we ran so far
            inc_sum_cost: float
                sum of runtimes of all incumbent runs

            Returns
            -------
            cutoff: int
                adapted cutoff
        '''

        if not self.run_obj_time:
            return self.cutoff

        # cost used by challenger for going over all its runs
        # should be subset of runs of incumbent (not checked for efficiency
        # reasons)
        chall_inst_seeds = run_history.get_runs_for_config(challenger)
        chal_sum_cost = sum_cost(config=challenger, instance_seed_pairs=chall_inst_seeds,
                                 run_history=run_history)
        cutoff = min(self.cutoff,
                     inc_sum_cost *
                     self.Adaptive_Capping_Slackfactor
                     - chal_sum_cost
                     )
        return cutoff

    def _compare_configs(self, incumbent: Configuration, 
                         challenger: Configuration, 
                         run_history: RunHistory,
                         aggregate_func: typing.Callable):
        '''
            compare two configuration wrt the runhistory 
            and return the one which performs better (or None if the decision is not safe)

            Decision strategy to return x as being better than y:
                1. x has at least as many runs as y
                2. x performs better than y on the intersection of runs on x and y

            Implicit assumption: 
                challenger was evaluated on the same instance-seed pairs as incumbent

            Parameters
            ----------
            incumbent: Configuration
                current incumbent
            challenger: Configuration
                challenger configuration
            run_history: RunHistory
                stores all runs we ran so far
            aggregate_func: typing.Callable
                aggregate performance across instances

            Returns
            -------
            None or better of the two configurations x,y
        '''

        inc_runs = run_history.get_runs_for_config(incumbent)
        chall_runs = run_history.get_runs_for_config(challenger)

        # performance on challenger runs
        chal_perf = aggregate_func(
            challenger, run_history, chall_runs)
        inc_perf = aggregate_func(
            incumbent, run_history, chall_runs)

        # Line 15
        if chal_perf > inc_perf and len(chall_runs) >= self.minR:
            # Incumbent beats challenger
            self.logger.debug("Incumbent (%.4f) is better than challenger (%.4f) on %d runs." % (
                inc_perf, chal_perf, len(chall_runs)))
            return incumbent

        # Line 16
        if len(chall_runs) >= len(inc_runs):
            # Challenger is as good as incumbent
            # and has the same number of runs
            # -> change incumbent

            n_samples = len(chall_runs)
            self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) on %d runs." % (
                chal_perf, inc_perf, n_samples))
            self.logger.info(
                "Changing incumbent to challenger: %s" % (challenger))
            self.stats.inc_changed += 1
            self.traj_logger.add_entry(train_perf=chal_perf,
                                       incumbent_id=self.stats.inc_changed,
                                       incumbent=challenger)
            return challenger

        return None  # undecided
