import copy
from collections import OrderedDict
import logging
import sys
import time
from collections import Counter

import numpy as np
 
from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib

from smac.smbo.objective import total_runtime, sum_cost
from smac.stats.stats import Stats
from smac.utils.constants import MAXINT

__author__ = "Katharina Eggensperger, Marius Lindauer"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Intensifier(object):

    '''
        takes challenger and incumbents and performs intensify
    '''

    def __init__(self, tae_runner, stats, traj_logger, rng, instances=None,
                 instance_specifics={}, cutoff=MAXINT, deterministic=False,
                 run_obj_time=True, run_limit=MAXINT, maxR=2000):
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
        '''
        self.stats = stats
        self.traj_logger = traj_logger
        # general attributes
        if instances is None:
            instances = []
        self.instances = set(instances)
        self.instance_specifics = instance_specifics
        self.logger = logging.getLogger("intensifier")
        self.run_limit = run_limit
        self.maxR = maxR
        self.rs = rng

        # scenario info
        self.cutoff = cutoff
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae_runner = tae_runner

        self.Adaptive_Capping_Slackfactor = 1.2

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")

    def intensify(self, challengers, incumbent, run_history, aggregate_func,
                  time_bound=MAXINT):
        '''
            running intensification to determine the incumbent configuration
            Side effect: adds runs to run_history

            Implementation of Procedure 2 in Hutter et al. (2011).

            Parameters
            ----------

            challengers : list of ConfigSpace.config
                promising configurations
            incumbent : ConfigSpace.config
                best configuration so far
            run_history : runhistory
                all runs on all instance,seed pairs for incumbent
            aggregate_func: func
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

        num_run = 0

        # Line 1 + 2
        for chall_indx, challenger in enumerate(challengers):
            if challenger == incumbent:
                self.logger.warning(
                    "Challenger was the same as the current incumbent; Skipping challenger")
                continue
            self.logger.debug("Intensify on %s", challenger)
            if hasattr(challenger, 'origin'):
                self.logger.debug(
                    "Configuration origin: %s", challenger.origin)
            inc_runs = run_history.get_runs_for_config(incumbent)

            # Line 3
            # First evaluate incumbent on a new instance
            if len(inc_runs) < self.maxR:
                # Line 4
                # find all instances that have the most runs on the inc
                inc_inst = [s.instance for s in inc_runs]
                inc_inst = list(Counter(inc_inst).items())
                inc_inst.sort(key=lambda x: x[1], reverse=True)
                max_runs = inc_inst[0][1]
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
                    status, cost, dur, res = self.tae_runner.start(config=incumbent,
                                                            instance=next_instance,
                                                            seed=next_seed,
                                                            cutoff=self.cutoff,
                                                            instance_specific=self.instance_specifics.get(next_instance, "0"))

                    num_run += 1
                else:
                    self.logger.debug(
                        "No further instance-seed pairs for incumbent available.")

            # Line 8
            N = 1

            inc_inst_seeds = set(run_history.get_runs_for_config(incumbent))
            inc_perf = aggregate_func(incumbent, run_history, inc_inst_seeds)

            # Line 9
            while True:
                chall_inst_seeds = set(map(lambda x: (
                    x.instance, x.seed), run_history.get_runs_for_config(challenger)))

                # Line 10
                missing_runs = list(inc_inst_seeds - chall_inst_seeds)

                # Line 11
                self.rs.shuffle(missing_runs)
                to_run = missing_runs[:min(N, len(missing_runs))]
                # Line 13 (Line 12 comes below...)
                missing_runs = missing_runs[min(N, len(missing_runs)):]

                inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))
                
                inc_sum_cost = sum_cost(config=incumbent, instance_seed_pairs=inst_seed_pairs,
                                        run_history=run_history)

                # Line 12
                for instance, seed in to_run:
                    # Run challenger on all <config,seed> to run
                    if self.run_obj_time:
                        chall_inst_seeds = set(map(lambda x: (
                            x.instance, x.seed), run_history.get_runs_for_config(challenger)))
                        chal_sum_cost = sum_cost(config=challenger, instance_seed_pairs=chall_inst_seeds,
                                                 run_history=run_history)
                        cutoff = min(self.cutoff,
                                     (inc_sum_cost - chal_sum_cost) *
                                     self.Adaptive_Capping_Slackfactor)

                        if cutoff < 0:  # no time left to validate challenger
                            self.logger.debug(
                                "Stop challenger itensification due to adaptive capping.")
                            break

                    else:
                        cutoff = self.cutoff

                    status, cost, dur, res = self.tae_runner.start(config=challenger,
                                                            instance=instance,
                                                            seed=seed,
                                                            cutoff=cutoff,
                                                            instance_specific=self.instance_specifics.get(instance, "0"))
                    num_run += 1

                # we cannot use inst_seed_pairs here since we could have less runs
                # for the challenger due to the adaptive capping
                chall_inst_seeds = set(map(lambda x: (
                    x.instance, x.seed), run_history.get_runs_for_config(challenger)))
                
                if len(chall_inst_seeds) < len(inst_seed_pairs):
                    #challenger got not enough runs because of exhausted config budget
                    self.logger.debug("No (or not enough) valid runs for challenger.")
                    break
                else:
                    chal_perf = aggregate_func(challenger, run_history, chall_inst_seeds)
                    inc_perf = aggregate_func(incumbent, run_history, chall_inst_seeds)
                    # Line 15
                    if chal_perf > inc_perf:
                        # Incumbent beats challenger
                        self.logger.debug("Incumbent (%.4f) is better than challenger (%.4f) on %d runs." % (
                            inc_perf, chal_perf, len(inst_seed_pairs)))
                        break
                    
                # Line 16
                if len(chall_inst_seeds) == len(inc_inst_seeds):
                    # Challenger is as good as incumbent -> change incumbent

                    n_samples = len(inst_seed_pairs)
                    self.logger.info("Challenger (%.4f) is better than incumbent (%.4f) on %d runs." % (
                        chal_perf / n_samples, inc_perf / n_samples, n_samples))
                    self.logger.info(
                        "Changing incumbent to challenger: %s" % (challenger))
                    incumbent = challenger
                    inc_perf = chal_perf
                    self.stats.inc_changed += 1
                    self.traj_logger.add_entry(train_perf=inc_perf,
                                               incumbent_id=self.stats.inc_changed,
                                               incumbent=incumbent)
                    break
                # Line 17
                else:
                    # challenger is not worse, continue
                    N = 2 * N

            if chall_indx >= 1 and num_run > self.run_limit:
                self.logger.debug(
                    "Maximum #runs for intensification reached")
                break
            elif chall_indx >= 1 and time.time() - self.start_time - time_bound >= 0:
                self.logger.debug("Timelimit for intensification reached ("
                                  "used: %f sec, available: %f sec)" % (
                                      time.time() - self.start_time, time_bound))
                break

        # output estimated performance of incumbent
        inc_runs = run_history.get_runs_for_config(incumbent)
        inc_perf = aggregate_func(incumbent, run_history, inc_runs)
        self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
            len(inc_runs), inc_perf))

        return incumbent, inc_perf
