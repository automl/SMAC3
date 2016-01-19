import copy
from collections import OrderedDict
import logging
import sys
import time
import random

import numpy

from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

MAXINT = 2**31 - 1


class Intensifier(object):
    '''
        takes challenger and incumbents and performs intensify
    '''

    def __init__(self, executor, challengers, incumbent, run_history, instances=None,
                 cutoff=MAXINT,
                 time_bound=MAXINT, run_limit=MAXINT, maxR=2000, rng=0):
        '''
        Constructor

        Parameters
        ----------
        executor : tae.executre_ta_run_*.ExecuteTARun* Object
            target algorithm run executor
        challengers : list of ConfigSpace.config
            promising configurations
        incumbent : ConfigSpace.config
            best configuration so far
        run_history : runhistory
            all runs on all instance,seed pairs for incumbent
        instances : list
            list of all instance ids
        time_bound : int
            time in [sec] available to perform intensify
        run_limit : int
            maximum number of runs
        maxR : int
            maximum number of runs per config
        '''
        # TODO It is probably important whether ta is deterministic

        # general attributes
        if instances is None:
            instances = []
        self.instances = set(instances)
        self.start_time = time.time()
        self.logger = logging.getLogger("intensifier")
        self.time_bound = time_bound
        self.run_limit = run_limit
        self.maxR = maxR
        self.rs = numpy.random.RandomState(rng)

        # info about current state
        self.challengers = challengers
        self.incumbent = incumbent
        self.run_history = run_history

        # scenario info
        self.cutoff = cutoff
        self.tae = executor

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")
        if self.time_bound <= 1:
            raise ValueError("time_bound must be > 1")

    def intensify(self):
        '''
            running intensification to determine the incumbent configuration
            Side effect: adds runs to run_history
        '''
        num_run = 0
        for challenger in self.challengers:
            self.logger.debug("Intensify on %s" %(challenger))
            inc_runs = self.run_history.get_runs_for_config(self.incumbent)
            # First evaluate incumbent on a new instance
            if len(inc_runs) < self.maxR:
                inc_scen = set([s[0] for s in inc_runs])
                next_seed = self.rs.randint(low=0, high=MAXINT,
                                            size=1)[0]
                next_instance = random.choice(
                    list((self.instances - inc_scen)))
                status, cost, dur, res = self.tae.run(config=self.incumbent,
                                                      instance=next_instance,
                                                      seed=next_seed,
                                                      cutoff=self.cutoff)
                self.run_history.add(config=self.incumbent,
                                     cost=cost, time=dur, status=status,
                                     instance_id=next_instance, seed=next_seed,
                                     additional_info=res)
                num_run += 1
            N = 1
            inc_inst_seeds = set(map(lambda x: (
                x.instance, x.seed), self.run_history.get_runs_for_config(self.incumbent)))

            while True:
                chall_inst_seeds = set(map(lambda x: (
                    x.instance, x.seed), self.run_history.get_runs_for_config(challenger)))

                missing_runs = list(inc_inst_seeds - chall_inst_seeds)

                self.rs.shuffle(missing_runs)
                to_run = missing_runs[:min(N, len(missing_runs))]
                missing_runs = missing_runs[min(N, len(missing_runs)):]

                for r in to_run:
                    # Run challenger on all <config,seed> to run
                    instance, seed = r.instance, r.seed
                    status, cost, dur, res = self.tae.run(config=challenger,
                                                          instance=instance,
                                                          seed=seed,
                                                          cutoff=self.cutoff)

                    self.run_history.add(config=challenger,
                                         cost=cost, time=dur, status=status,
                                         instance_id=instance, seed=seed,
                                         additional_info=res)
                    num_run += 1

                challenger_runs = self.run_history.get_runs_for_config(
                    challenger)
                chal_inst_seeds = map(lambda x: (
                    x.instance, x.seed), self.run_history.get_runs_for_config(challenger))
                chal_perf = sum(map(lambda x: x.cost, challenger_runs))

                inc_id = self.run_history.config_ids[self.incumbent.__repr__()]
                inc_perfs = []
                for i, r in chal_inst_seeds:
                    inc_k = self.run_history.RunKey(inc_id, i, r)
                    inc_perfs.append(self.run_history.data[inc_k].cost)
                inc_perf = sum(inc_perfs)

                if chal_perf > inc_perf:
                    # Incumbent beats challenger
                    self.logger.debug("Incumbent is better than challenger.")
                    break
                elif len(missing_runs) == 0:
                    # Challenger is as good as incumbent -> change incu
                    self.logger.debug("Changing incumbent to challenger")
                    self.incumbent = challenger
                    break
                else:
                    # challenger is not worse, continue
                    N = 2 * N

                if num_run > self.run_limit:
                    self.logger.debug(
                        "Maximum #runs for intensification reached")
                    break
                elif time.time() - self.start_time - self.time_bound > 0:
                    self.logger.debug("Timelimit for intensification reached")
                    break

        return self.incumbent