import time
import logging

__author__ = "Marius Lindauer"
__copyright__ = "Copyright 2016, ML4AAD"
__license__ = "3-clause BSD"
__maintainer__ = "Marius Lindauer"
__email__ = "lindauer@cs.uni-freiburg.de"
__version__ = "0.0.1"


class Stats(object):

    '''
        all statistics collected during configuration run
    '''
    
    def __init__(self, scenario):
          
        self.__scenario = scenario
        
        self.ta_runs = 0
        self.wallclock_time_used = 0
        self.ta_time_used = 0
        self.inc_changed = 0
    
        self._start_time = None
        self._logger = logging.getLogger("Stats")

    def start_timing(self):
        '''
            starts the timer (for the runtime configuration budget)
        '''
        if self.__scenario:
            self._start_time = time.time()
        else:
            raise ValueError("Scenario is missing")

    def get_used_wallclock_time(self):
        '''
            returns used wallclock time
            Returns
            -------
            wallclock_time : int
                used wallclock time in sec
        '''
        
        return time.time() - self._start_time

    def get_remaing_time_budget(self):
        '''
            subtracts the runtime configuration budget with the used wallclock time
        '''
        if self.__scenario:
            return self.__scenario.wallclock_limit - (time.time() - self._start_time)
        else:
            raise "Scenario is missing"

    def get_remaining_ta_runs(self):
        '''
           subtract the target algorithm runs in the scenario with the used ta runs 
        '''
        if self.__scenario:
            return self.__scenario.ta_run_limit - self.ta_runs
        else:
            raise "Scenario is missing"

    def get_remaining_ta_budget(self):
        '''
            subtracts the ta running budget with the used time
        '''
        if self.__scenario:
            return self.__scenario.algo_runs_timelimit - self.ta_time_used

    def print_stats(self):
        '''
            prints all statistics
        '''
        self._logger.info("##########################################################")
        self._logger.info("Statistics:")
        self._logger.info("#Target algorithm runs: %d" %(self.ta_runs))
        self._logger.info("Used wallclock time: %.2f sec" %(time.time() - self._start_time))
        self._logger.info("Used target algorithm runtime: %.2f sec" %(self.ta_time_used))
        
        self._logger.info("##########################################################")    
