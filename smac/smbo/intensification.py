import copy
from collections import OrderedDict
import logging
import sys
import time
import random
from collections import Counter

import numpy

from smac.tae.execute_ta_run_aclib import ExecuteTARunAClib
from smac.utils.io.traj_logging import TrajLogger
from smac.stats.stats import Stats

__author__ = "Katharina Eggensperger, Marius Lindauer"
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
                 cutoff=MAXINT, deterministic=False, run_obj_time=True,
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
        cutoff : int
            runtime cutoff of TA runs
        deterministic: bool
            whether the TA is deterministic or not
        run_obj_time: bool
            whether the run objective is runtime or not (if true, apply adaptive capping)
        time_bound : int
            time in [sec] available to perform intensify
        run_limit : int
            maximum number of runs
        maxR : int
            maximum number of runs per config
        '''
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
        self.deterministic = deterministic
        self.run_obj_time = run_obj_time
        self.tae = executor
        
        self.trajLogger = TrajLogger()
        self.Adaptive_Capping_Slackfactor = 1.2

        if self.run_limit < 1:
            raise ValueError("run_limit must be > 1")
        if self.time_bound <= 1:
            raise ValueError("time_bound must be > 1")

    def intensify(self):
        '''
            running intensification to determine the incumbent configuration
            Side effect: adds runs to run_history
            
            Returns
            -------
            incumbent: Configuration()
                current (maybe new) incumbent configuration
        '''
        num_run = 0
        for challenger in self.challengers:
            self.logger.debug("Intensify on %s" % (challenger))
            inc_runs = self.run_history.get_runs_for_config(self.incumbent)
            # First evaluate incumbent on a new instance
            if len(inc_runs) <= min(self.maxR, len(self.instances)):
                # find all instances that have the most runs on the inc
                inc_inst = [s.instance for s in inc_runs]
                inc_inst = list(Counter(inc_inst).items())
                inc_inst.sort(key=lambda x: x[1], reverse=True)
                max_runs = inc_inst[0][1]
                inc_inst = set(map(lambda x: x[0], filter(lambda x: x[1]==max_runs, inc_inst)))
                
                if self.deterministic:
                    next_seed = 0
                else:
                    next_seed = self.rs.randint(low=0, high=MAXINT,
                                            size=1)[0]
                                          
                available_insts = (self.instances - inc_inst)
                if available_insts: 
                    next_instance = random.choice(list(available_insts))
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
                
                inst_seed_pairs = list(inc_inst_seeds - set(missing_runs))
                print("Incumbent perfs:")
                inc_perf, inc_time = self.get_perf_and_time(self.incumbent, inst_seed_pairs)
                
                _, chal_time = self.get_perf_and_time(challenger, chall_inst_seeds) 
                #TODO: do we have to consider PAR10 here instead of PAR1?
                
                for instance, seed in to_run:
                    # Run challenger on all <config,seed> to run
                    if self.run_obj_time:
                        cutoff = min(self.cutoff, (inc_perf - chal_time) * self.Adaptive_Capping_Slackfactor)
                        #print("Adaptive Capping cutoff: %f" %(cutoff))
                    else:
                        cutoff = self.cutoff
                    status, cost, dur, res = self.tae.run(config=challenger,
                                                          instance=instance,
                                                          seed=seed,
                                                          cutoff=cutoff)

                    self.run_history.add(config=challenger,
                                         cost=cost, time=dur, status=status,
                                         instance_id=instance, seed=seed,
                                         additional_info=res)
                    num_run += 1

                print("Challenger perfs:")
                chal_perf, chal_time = self.get_perf_and_time(challenger, inst_seed_pairs)

                if chal_perf > inc_perf:
                    # Incumbent beats challenger
                    self.logger.debug("Incumbent (%.2f) is better than challenger (%.2f) on %d runs." %(inc_perf, chal_perf, len(inst_seed_pairs)))
                    break
                elif len(missing_runs) == 0:
                    # Challenger is as good as incumbent -> change incu
                    
                    self.logger.info("Challenger (%.2f) is better than incumbent (%.2f) on %d runs." %(chal_perf, inc_perf, len(inst_seed_pairs)))
                    self.logger.info(
                        "Changing incumbent to challenger: %s" % (challenger))
                    self.incumbent = challenger
                    Stats.inc_changed += 1
                    self.trajLogger.add_entry(train_perf=chal_perf, 
                                              incumbent_id=Stats.inc_changed, 
                                              incumbent=challenger)
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

        # output estimated performance of incumbent
        inc_runs = self.run_history.get_runs_for_config(self.incumbent)
        inc_perf = numpy.mean(map(lambda x: x.cost, inc_runs))
        self.logger.info("Updated estimated performance of incumbent on %d runs: %.4f" % (
            len(inc_runs), inc_perf))

        return self.incumbent
    
    def get_perf_and_time(self, config, inst_seeds):
            '''
            returns perf and used runtime of a configuration
    
            Parameters
            ----------
            config: Configuration()
                configuration to get stats for
            inst_seeds: list
                list of tuples of instance-seeds pairs
                
            Returns
            ----------
            perf : float 
                sum of cost values in runhistory
            time: float
                sum of time values in runhistory
            '''
        
            try:
                id_ = self.run_history.config_ids[config.__repr__()]
            except KeyError: # challenger was not running so far
                return MAXINT, 0
            perfs = []
            times = []
            for i, r in inst_seeds:
                k = self.run_history.RunKey(id_, i, r)
                perfs.append(self.run_history.data[k].cost)
                times.append(self.run_history.data[k].time)
            print(perfs)
            print(times)
            perf = sum(perfs)
            time = sum(times)
            
            return perf, time