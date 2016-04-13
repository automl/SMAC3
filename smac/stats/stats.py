import time
import logging

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "GPLv3"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Stats(object):

    '''
        all statistics collected during configuration run
        This is a static class without initialization to get an easy access every in the code
    '''
    
    scenario = None
    
    ta_runs = 0
    wallclock_time_used = 0
    incumbent_changed = 0
    ta_time_used = 0
    inc_changed = 0

    _start_time = None
    _logger = logging.getLogger("Stats")

    @staticmethod
    def start_timing():
        '''
            starts the timer (for the runtime configuration budget)
        '''
        if Stats.scenario:
            Stats._start_time = time.time()
        else:
            raise ValueError("Scenario is missing")

    @staticmethod
    def get_used_wallclock_time():
        '''
            returns used wallclock time
            Returns
            -------
            wallclock_time : int
                used wallclock time in sec
        '''
        
        return time.time() - Stats._start_time

    @staticmethod
    def get_remaing_time_budget():
        '''
            subtracts the runtime configuration budget with the used wallclock time
        '''
        if Stats.scenario:
            return Stats.scenario.wallclock_limit - (time.time() - Stats._start_time)
        else:
            raise "Scenario is missing"

    @staticmethod
    def get_remaining_ta_runs():
        '''
           subtract the target algorithm runs in the scenario with the used ta runs 
        '''
        if Stats.scenario:
            return Stats.scenario.ta_run_limit - Stats.ta_runs
        else:
            raise "Scenario is missing"

    @staticmethod
    def get_remaining_ta_budget():
        '''
            subtracts the ta running budget with the used time
        '''
        if Stats.scenario:
            return Stats.scenario.algo_runs_timelimit - Stats.ta_time_used

    @staticmethod
    def print_stats():
        '''
            prints all statistics
        '''
        Stats._logger.info("##########################################################")
        Stats._logger.info("Statistics:")
        Stats._logger.info("#Target algorithm runs: %d" %(Stats.ta_runs))
        Stats._logger.info("Used wallclock time: %.2f sec" %(time.time() - Stats._start_time))
        Stats._logger.info("Used target algorithm runtime: %.2f sec" %(Stats.ta_time_used))
        
        Stats._logger.info("##########################################################")    
