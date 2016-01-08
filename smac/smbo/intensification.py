import copy
from collections import OrderedDict
import logging
import sys
import time

import numpy

from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib

__author__ = "Katharina Eggensperger"
__copyright__ = "Copyright 2015, ML4AAD"
__license__ = "BSD"
__maintainer__ = "Katharina Eggensperger"
__email__ = "eggenspk@cs.uni-freiburg.de"
__version__ = "0.0.1"

MAXINT = 2**31-1


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
        num_run = 0
        for challenger in self.challengers:
            inc_runs = self.run_history.get_runs_for_config(self.incumbent)
            # First evaluate incumbent on a new instance
            if len(inc_runs) < self.maxR:
                inc_scen = set([s[0] for s in inc_runs])
                # TODO sample instance where inc has not yet been evaluated
                next_seed = self.rs.randint(low=0, high=MAXINT,
                                                   size=1)[0]
                next_instance = sorted(list((self.instances - inc_scen)))
                next_instance = next_instance[next_seed % len(next_instance)]
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
            while True:
                # TODO get a list of all seeds where challenger has not been evaluated, but incumbent
                # TODO get a list of all instances where challenger has not been evaluated, but incumbent
                challenger_runs = self.run_history.get_runs_for_config(challenger)
                # build a boolean matrix
                challenger_runs_matrix = numpy.zeros([])
                # TODO: here I stopped, fill matrix with ones where inc has been evaluated and create missing_runs list
                self.rs.shuffle(missing_runs)
                to_run = missing_runs[:min(N, len(missing_runs))]
                missing_runs = missing_runs[min(N, len(missing_runs)):]
                for r in to_run:
                    # Run challenger on all <config,seed> to run
                    seed, instance = r
                    status, cost, dur, res = self.tae.run(config=challenger,
                                                           instance=instance,
                                                           seed=seed,
                                                           cutoff=self.cutoff)
                    self.run_history.add(config=challenger,
                                         cost=cost, time=dur, status=status,
                                         instance_id=instance, seed=seed,
                                         additional_info=res)
                    num_run += 1
                incu_perf = self.runhistory.performance(self.incumbent)
                chal_perf = self.runhistory.performance(challenger)
                if chal_perf > incu_perf:
                    # Incumbent beats challenger
                    break
                elif len(missing_runs) == 0:
                    # Challenger is as good as incumbent -> change incu
                    self.incumbent = challenger
                else:
                    # challenger is not worse, continue
                    N = 2*N

                if num_run > self.run_limit:
                    self.logger("Maximum #runs for intensification reached")
                    break
                elif time.time() - self.start_time - self.time_bound < 0:
                    self.logger("Timelimit for intensification reached")
                    break

        return self.incumbent, self.run_history





